import os
import py_compile
import time
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
from loguru import logger
from torch.utils.data import DistributedSampler

import dataloader
import wandb
from arguments import parse_args
from models.enums import Split, TrainingSample, float_pt
from models.trainer import Trainer
from utils import pytorch_utils as util

# from utils.ddp_utils import EXIT

py_compile.compile("train_eval.py")
torch.multiprocessing.set_sharing_strategy("file_system")

warnings.filterwarnings("ignore")
dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5,6"


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def set_distributed(rank, world_size, args):
    args.master_port = int(os.environ.get("MASTER_PORT", args.master_port))
    args.master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    if rank == 0:
        logger.info(f"{args.master_addr=} {args.master_port=}")
    tcp_store = torch.distributed.TCPStore(args.master_addr, args.master_port,
                                           world_size, rank == 0)
    torch.distributed.init_process_group('nccl',
                                         store=tcp_store,
                                         rank=rank,
                                         world_size=world_size)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
    else:
        assert world_size == 1
        device = torch.device("cpu")
    args.device = device


def reduce_dict(input_dict, world_size):
    """
    Args:
        input_dict (dict): all the values will be reduced
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        for k in input_dict.keys():
            names.append(k)
            if type(input_dict[k]) is dict:
                values.append(reduce_dict(input_dict[k], world_size))
            else:
                torch.distributed.all_reduce(input_dict[k])
                values.append(input_dict[k] / world_size)
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def gather_dict(input_dict, world_size):
    """
    Args:
        input_dict (dict): all the values will be gathered
    Gather the values in the dictionary from all processes so that all processes
    have the same results. Returns a dict with the same fields as
    input_dict, after gathered.
    """
    if world_size < 2:
        return input_dict
    # with torch.inference_mode():
    names = []
    values = []
    for k in input_dict.keys():
        names.append(k)
        if type(input_dict[k]) is dict:
            values.append(gather_dict(input_dict[k], world_size))
        else:

            tensor_placeholder = [
                torch.ones_like(input_dict[k]) for _ in range(world_size)
            ]
            torch.distributed.all_gather(tensor_placeholder, input_dict[k])
            values.append(tensor_placeholder)

    gathered_dict = {k: v for k, v in zip(names, values)}
    return gathered_dict


def get_batch(data, i, prev_state, future_feat) -> TrainingSample:
    # Returns all variables in the batch in tensor format.
    # i for the timestamp
    poses = float_pt([t.numpy() for t in data["poses"][i]])
    p_poses = float_pt([t.numpy() for t in data["p_poses"][i]])
    rt_c = float_pt([t.numpy() for t in data["extrinsic"][i]])
    rt_p = (float_pt([t.numpy() for t in data["extrinsic"][i - 1]])
            if i > 1 else torch.eye((4)).repeat(rt_c.shape[0], 1, 1))

    bbox1 = data["bbox"][i]  # BxNx4
    bbox1_p = data["p_bbox"][i]  # BxNx4

    batch: TrainingSample = {
        "images": data["image"][i],
        "cls_indices": data["cls_indices"][i],
        "intrinsic": data["intrinsic"][i],
        "label": data["label"][i],
        "depth": data["depth"][i],
        "poses": poses,
        "extrinsic_curr": rt_c,
        "extrinsic_prev": rt_p,
        "posecnn_poses": p_poses,
        "bbox": bbox1,
        "posecnn_bbox": bbox1_p,
        "prev_state": prev_state,
        "timestep": torch.tensor(i),
        "future_feat": future_feat,
    }
    return batch


def solve(
    rank,
    world_size,
    args,
):
    set_distributed(rank, world_size, args)

    trainer = Trainer(args)
    loaders = get_dataloader(args, world_size)
    trainer.initialisation(rank)
    start_epoch = 0
    start_epoch, is_partially_trained = trainer.try_init_trainer(rank)

    if rank == 0:
        summary_writer = trainer.writer
        wandb.watch(trainer.model)
        total_params = sum(p.numel() for p in trainer.model.parameters()
                           if p.requires_grad)
        logger.info("Total number of parameters {}".format(total_params))
        summary_writer.log_text("Arguments", "{0} <br> ".format(args))

    last_saved_time = datetime.now()
    if "train" not in args.split:
        args.num_epochs = 1

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        for split, loader in loaders.items():
            trainer.on_epoch_start(epoch, split)
            if split not in args.split:
                continue
            with torch.set_grad_enabled(split == Split.Train.value):
                t0 = time.time()
                t_loader = time.time()

                for batch_idx, batch in enumerate(loader):
                    args.split_dataset_size = len(loader)
                    trainer.zero_grad()
                    if rank == 0:
                        logger.info(
                            f"Got {batch_idx+1}/{len(loader)} batches in {split} epoch {epoch} in {time.time() - t_loader} seconds"
                        )
                        logger.info("Forward pass")
                    t0 = time.time()
                    loss, outputs = trainer.forward_impl(batch, rank)

                    torch.distributed.all_reduce(loss)
                    loss = loss / world_size
                    outputs = gather_dict(outputs, world_size)

                    output = outputs["pose_out"]
                    if rank == 0:
                        t1 = time.time()
                        logger.info(
                            f"Done with forward pass in {t1-t0} seconds")
                    if split == Split.Evaluate.value:
                        # Uses posecnn bbox as the ROI to get more accurate estimations. Also saves output for every iteration.
                        is_keyframe = batch["is_keyframe"]
                        fl = batch["file_indices"]
                        batch_p = batch
                        if args.use_posecnn:
                            batch_p["bbox"] = batch["posecnn_bbox"]
                        loss_p, outputs_p = trainer.forward_impl(batch_p, rank)
                        outputs_p = gather_dict(outputs_p, world_size)
                        output_p = outputs_p["pose_out"]
                        trainer.on_iteration_complete_eval(
                            output, output_p, batch, is_keyframe, fl)
                        if rank == 0:
                            logger.info(
                                "Keyframe Evaluation (batch index/Total) ({0}/{1})"
                                .format(batch_idx, len(loader)))

                    if split != Split.Train.value:
                        continue
                    trainer.zero_grad()
                    if rank == 0:
                        logger.info("Backward pass")
                    loss.backward()
                    if rank == 0:
                        logger.info("Done backward pass")
                    trainer.step()
                    if rank == 0:
                        logger.info(
                            "{0}ing Loss average:{1} batch index/total {2}/{3} epoch {4} in time: {5}"
                            .format(
                                split,
                                loss,
                                batch_idx,
                                len(loader),
                                epoch,
                                (time.time() - t0),
                            ))
            if rank == 0:
                logger.info(f"{split} has finished")
                total_time = time.time() - t0
                logger.info("Average time taken: {0}".format(
                    total_time / (len(loader) * 20)))
                if split == Split.Train.value:
                    trainer.save_checkpoint()
        if rank == 0:
            last_saved_time = trainer.on_epoch_complete()


def get_dataloader(args, world_size):
    Transform = util.Transform()
    dataloader_class = dataloader.VideoLoader
    train_file = os.path.join(args.data_root_path, args.train_file)
    val_file = os.path.join(args.data_root_path, args.val_file)
    collate_fn = dataloader.collate_fn

    train_dset = dataloader_class(
        args.data_root_path,
        train_file,
        transform=Transform(),
        add_noise=True,
        add_rot=True,
        add_translation=True,
        roi_noise=args.roi_noise,
        add_jitter=args.add_jitter,
        video_length=args.video_length,
        step=args.step,
        is_train=True,
    )

    val_dset = dataloader_class(
        args.data_root_path,
        val_file,
        transform=Transform(),
        video_length=args.video_length,
        step=args.step,
        is_train=False,
    )

    eval_dset = dataloader_class(
        args.data_root_path,
        os.path.join(args.data_root_path, args.keyframe_file),
        transform=Transform(),
        video_length=args.video_length,
        step=args.step,
        is_train=False,
    )

    loaders = {}
    if "train" in args.split:
        train_dataloader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size // world_size,
            sampler=DistributedSampler(train_dset),
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context='fork')
        loaders["train"] = train_dataloader

    if "test" in args.split:
        val_dataloader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size // world_size,
            sampler=DistributedSampler(val_dset, shuffle=False),
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context='fork')
        loaders["test"] = val_dataloader

    if "eval" in args.split:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dset,
            batch_size=args.batch_size // world_size,
            sampler=DistributedSampler(eval_dset, shuffle=False),
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
            multiprocessing_context='fork')
        loaders["eval"] = eval_dataloader

    return loaders


if __name__ == "__main__":
    args = parse_args()

    data_root_path = dir_path + "/data/YCB"
    args.data_root_path = data_root_path
    _, points = dataloader.load_object_points(data_root_path)
    points = torch.from_numpy(points)
    classes = dataloader.get_classes(data_root_path)
    args.keyframe_list = dataloader.get_keyframe_list(data_root_path)

    np.random.seed(5)
    torch.manual_seed(500)
    torch.backends.cudnn.deterministic = True
    torch.autograd.set_detect_anomaly(True)
    device = "cpu"
    args.root_path = dir_path

    start_time = datetime.now()
    logger.info("Time: {0}".format(start_time))

    # Train the model
    logger.info("Starting training for {} epoch(s)".format(args.num_epochs))
    epoch_start = 1

    cuda_device_cnt = torch.cuda.device_count()
    world_size = args.num_gpu if args.num_gpu > 0 else cuda_device_cnt
    args.master_port = find_free_port()
    mp.spawn(solve, args=(world_size, args), nprocs=world_size, join=True)
    logger.info("Finished")
