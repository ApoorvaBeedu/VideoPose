import copy
import datetime
import json
import os

import cv2
import dataloader
import numpy as np
import scipy.io as scio
import torch
import torch.multiprocessing
import torch.optim as optim
import wandb
import yaml
from click import echo
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import models
from models.enums import LossDict, Losses, Split, TrainingSample
from utils import pytorch_utils as util
from utils.ddp_utils import convert_groupnorm_model
from utils.drawing_utils import DrawFromPose


def get_model(args):
    return models.VideoPose(args)


def visualise(output,
              batch,
              points,
              classes,
              output_ind=-1):
    """
    This function is a utility to visualise results.
    """

    try:
        draw
    except NameError:
        draw = DrawFromPose()
    try:
        unnormalise
    except NameError:
        unnormalise = util.Transform().unnormalise()

    temp = batch["image"][output_ind, -1, :, :, :]
    cls_indices = batch["cls_indices"][output_ind, -1]
    K = batch["intrinsic"][output_ind, -1]
    bbox1 = batch["bbox"][output_ind, -1]
    pose_gt = batch["poses"][output_ind, -1]


    temp = unnormalise(temp).data.cpu().numpy()
    bbox1 = bbox1.float()
    K = K.float()
    output = output.cpu()

    p_est = output[output_ind, -1, :, :].float()
    plot = draw.drawModel(
        temp,
        pose_gt,
        p_est.detach(),
        bbox1,
        K.cpu().numpy(),
        points,
        cls_indices,
        classes,
    )
    return plot


def format_dictloss(split, losses):
    """
    Prints all the losses.
    """
    # Logging stuff, printing all the individual losses
    loss_string = "".join(
        ["{: >32}: {}\n".format(k, v) for k, v in losses.items()])
    logger.info("{0} losses:\n{1}".format(split, loss_string))


def writer_log_text(phase, optimizer, epoch, train_loss, val_loss, train_rt,
                    val_rt, writer):
    lr_string = "{0} <br> lr: ".format(phase)
    for param_group in optimizer.param_groups:
        logger.info("Current lr: {0}".format(param_group["lr"]))
        lr_string += "{0} ".format(param_group["lr"])
    writer.log_text(
        "{0}_losses".format(phase),
        "{0} <br> epoch: {1} <br> train_rot_loss: {2} "
        "<br> validation_rot_loss: {3} <br> train_rt_loss:"
        " {4} <br> val_rt_loss: {5}".format(lr_string, epoch, train_loss,
                                            val_loss, train_rt, val_rt),
    )


class GenericTrainer:
    def __init__(self):
        super(GenericTrainer).__init__()

        self.model = torch.nn.Module
        self.optimizer = None
        self.lr_scheduler = None

    def model(self) -> torch.nn.Module:
        return self.model

    def writer(self) -> util.WriteLogs():
        if not self.writer:
            return None
        return self.writer

    def initialisation(self):
        raise NotImplementedError

    def try_init_trainer(self):
        raise NotImplementedError

    def step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def zero_grad(self):
        self.model.zero_grad()
        self.optimizer.zero_grad()

    def save_checkpoint(self):
        raise NotImplementedError

    def on_epoch_start(self):
        raise NotImplementedError

    def on_epoch_complete(self):
        raise NotImplementedError

    def forward_impl(self):
        raise NotImplementedError


class Trainer(GenericTrainer):
    def __init__(self, args) -> None:
        super().__init__()

        model = get_model(args)
        self.optimizer = util.get_optimizer(args, model.get_params(args.lr))
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=args.lr_multiplier,
            patience=3,
            threshold=0.1,
            verbose=True,
        )
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.args = args
        self.loss_min = np.inf

        _, points = dataloader.load_object_points(args.data_root_path,
                                                  dense=True)
        self.points = torch.from_numpy(points)
        self.classes = dataloader.get_classes(args.data_root_path)
        self.losses = Losses

    def initialisation(self, rank):
        dir_path = self.args.root_path
        args = self.args
        start_time = datetime.datetime.now()
        time_now = start_time.strftime("%d%m%Y_%H%M%S")
        if rank == 0:
            tb_writer = SummaryWriter(log_dir=args.writer_dir + "/" + time_now)
            wandb.init(project=args.env_name, entity=self.args.entity)
            wandb.config.update(args)
            writer = util.WriteLogs(tb_writer, wandb)
            self.writer = writer

            args.wandb_name = wandb.run.name
            if "eval" in self.args.split:
                path = os.path.join(dir_path, "evaluation_results_video",
                                    args.env_name, args.wandb_name)
                eval_video_path = os.path.join(dir_path,
                                               "evaluation_results_video")
                if not os.path.isdir(
                        os.path.join(eval_video_path, args.env_name)):
                    os.mkdir(os.path.join(eval_video_path, args.env_name))
                if not os.path.isdir(path):
                    os.mkdir(path)
                self.evaluate_save_dir = path

                path_o = os.path.join(dir_path, "output", args.env_name,
                                      args.wandb_name)
                if not os.path.isdir(
                        os.path.join(dir_path, "output", args.env_name)):
                    os.mkdir(os.path.join(dir_path, "output", args.env_name))
                if not os.path.isdir(path_o):
                    os.mkdir(path_o)
                self.output_save_dir = path_o

            if not os.path.isdir(os.path.join(dir_path, "logs")):
                os.mkdir(os.path.join(dir_path, "logs"))

            if not os.path.isdir(args.model_dir):
                os.mkdir(args.model_dir)
            args.model_dir = os.path.join(args.model_dir, args.wandb_name)
            if not os.path.isdir(args.model_dir):
                os.mkdir(args.model_dir)
            logger.start(args.model_dir + "/bash_logger.log")

            self.write_args_to_json()

    def write_args_to_json(self):
        temp_args = copy.deepcopy(self.args)
        del temp_args.device
        del temp_args.keyframe_list
        with open(temp_args.model_dir + "/arguments.json", "w") as f:
            json.dump(temp_args.__dict__, f)
        del temp_args

    def try_init_trainer(
        self,
        rank,
    ) -> int:
        args = self.args
        is_part_ckpt = 0
        epoch_start = 0
        if "posecnn" in self.args.backbone:
            util.load_from_tensorflow(self.model)
        self.model = convert_groupnorm_model(self.model)
        self.model = self.model.to(args.device)
        self.model = DDP(self.model,
                         device_ids=[args.device],
                         output_device=args.device,
                         find_unused_parameters=True)

        if rank == 0:
            if args.restore_file is not None and args.restore_file != "None":
                if args.restart_optimiser:
                    self.model, _, epch_chk, is_part_ckpt = util.restore_from_file(
                        args, self.model, None, rank)
                else:
                    (
                        self.model,
                        self.optimizer,
                        epch_chk,
                        is_part_ckpt,
                    ) = util.restore_from_file(args, self.model,
                                               self.optimizer, rank)
                if is_part_ckpt > 0:
                    epoch_start = epch_chk
                else:
                    epoch_start = epch_chk + 1

        return epoch_start, is_part_ckpt

    def step(self):
        self.optimizer.step()
        if self.args.use_scheduler:
            metric = self.losses.train.rt_loss()
            self.lr_scheduler.step(metric)

    def on_epoch_start(self, epoch, split) -> None:
        # Clear the test losses
        self.losses.test = LossDict()

        # Update epoch and split self variables
        self.epoch = epoch
        self.split = split
        pass

    def save_checkpoint(self, is_partial=0) -> float:
        args = self.args
        util.save_checkpoint(
            {
                "epoch": self.epoch,
                "state_dict": self.model.state_dict(),
                "optim_dict": self.optimizer.state_dict(),
                "is_partial": is_partial,
            },
            checkpoint=args.model_dir,
            counter="{0}".format("%04d" % self.epoch),
        )
        return datetime.datetime.now()

    def on_epoch_complete(self) -> None:
        logger.info("epoch completed")
        args = self.args
        test_losses = self.losses.test
        loss_val_rt = test_losses.get_rot_loss()
        time = self.save_checkpoint()

        # Logging test losses to wandb
        loss_dict = {
            "rotation_distance_loss": loss_val_rt,
            "rt_loss": test_losses.get_rt_loss(),
        }
        self.writer.log_scalars_dict("test Losses", loss_dict,
                                     test_losses.loss_avg.steps)

        if loss_val_rt < self.loss_min:
            self.loss_min = loss_val_rt
            stream = open(os.path.join(args.model_dir, "best_checkpoint.yaml"),
                          "w")
            dict = {
                "model": os.path.basename(args.model_dir),
                "epoch": self.epoch,
                "partial_epoch": 0,
                "Accuracy RT": self.loss_min,
                "Accuracy Rot": loss_val_rt
            }
            yaml.dump(dict, stream)

        writer_log_text(
            self.split,
            self.optimizer,
            self.epoch,
            self.losses.train.get_rot_loss(),
            self.losses.test.get_rot_loss(),
            self.losses.train.get_rt_loss(),
            self.losses.test.get_rt_loss(),
            self.writer,
        )
        return time

    def on_iteration_complete(self, output, batch, split_losses) -> None:
        writer = self.writer
        poses = batch["poses"]
        epoch = self.epoch
        counter = split_losses.loss_avg.steps
        if self.split == Split.Test.value:
            counter = epoch
        split = self.split
        output_losses = output["output_losses"].copy()
        for k, v in output_losses.items():
            output_losses[k] = v.mean().item()

        split_losses.rotation_distance_loss.update(
            output_losses["rotation_distance_loss"])
        split_losses.rt_loss.update(output_losses["rt_loss"])

        plot = visualise(output["pose_out"], batch, self.points, self.classes)
        if counter % (self.args.split_dataset_size // 10) == 0:
            writer.log_images(f"{split}_projection", plot, step=0)

        # prints losses
        format_dictloss(
            f"{split}ing",
            output_losses,
        )
        # Prints few samples
        logger.info("\noutput: {0}\ninput {1}\n".format(
            output["pose_out"][0, 0, 0, :],
            torch.cat((models.mat2quat(poses[0, 0, 0, :,
                                             0:3]), poses[0, 0, 0, :, 3])),
        ))

        if split == Split.Train.value:
            writer.log_scalar(
                f"{split} loss_avg",
                split_losses.loss_avg(),
                counter,
            )
            writer.log_scalars_dict(f"{split} Losses", output_losses, counter)

    def on_iteration_complete_eval(self, outputs, batch, is_key,
                                   filenames) -> None:
        # Goes over objects in the batch.
        # NOTE: This assumes a single batch size!!!
        for b_idx, output in enumerate(outputs):
            plot = visualise(
                output.unsqueeze(0),
                batch,
                self.points,
                self.classes,
                output_ind=-1,
            ).transpose(1, 2, 0)
            fl = f"{filenames[b_idx, -1][0]}_{filenames[b_idx,-1][1]}"
            filename = "{0}/{1}".format("%04d" % filenames[b_idx, -1][0],
                                        "%06d" % filenames[b_idx, -1][1])
            ind = self.args.keyframe_list.index(filename)
            cv2.imwrite(
                "{0}/keyframe_video_{1}.png".format(self.output_save_dir, fl),
                cv2.cvtColor(plot, cv2.COLOR_RGB2BGR))
            try:
                scio.savemat(
                    "{0}/{1}.mat".format(self.evaluate_save_dir, "%04d" % ind),
                    {
                        "poses": output.cpu().data.numpy(),
                        "filenames": filename,
                        "cls_indices":
                        [batch["cls_indices"][b_idx, -1].numpy()],
                    },
                )
            except:
                print("some error")

    def forward_impl(
        self,
        batch,
        rank,
    ):

        # Settimg train/eval mode
        self.model.train()
        if self.split != (Split.Train.value):
            self.model.eval()

        # Forward pass
        output = self.model(batch, rank)

        # Gather loss
        loss = output["loss_value"]

        # Create output and update losses for the current split
        output["pose_out"] = torch.cat(
            (output["_R"].detach(), output["_T"].detach()), dim=3)

        if "train" in self.split:
            split_losses = self.losses.train
        elif "test" in self.split:
            split_losses = self.losses.test
        else:
            split_losses = self.losses.eval

        split_losses.loss_avg.update(loss.item())
        if self.split != (Split.Evaluate.value) and rank == 0:
            self.on_iteration_complete(output, batch, split_losses)

        return loss, output
