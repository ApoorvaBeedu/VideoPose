import argparse
import json
import os
import os.path as osp


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2list(v):
    if isinstance(v, str):
        return v.split()


def parse_args():
    parser = argparse.ArgumentParser()
    # checkpoint and model directories
    parser.add_argument(
        "--model_dir",
        default="./model",
        help=
        "Optional, path to the model directory containing weights to reload before training",
    )
    parser.add_argument(
        "--restore_file",
        default=None,
        help=
        "Optional, name of the file in --model_dir containing weights to reload before training",
    )
    parser.add_argument(
        "--pretrained_path",
        default=None,
        help="path for the pretrained weights for the single frame prediction",
    )
    parser.add_argument(
        "--depth_model_path",
        default=None,
        help="complete path to the pretrained depth model",
    )

    # Visdom and wandb related
    parser.add_argument(
        "--env_name",
        type=str,
        default="VideoPose_NeuRIPS",
    )
    parser.add_argument(
        "--entity",
        type=str,
    )

    # Optimiser related
    parser.add_argument(
        "--optimiser",
        type=str,
        default="adam",
        help="adam or sgd",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate for the optimiser",
    )
    parser.add_argument(
        "--pretrained_lr",
        default=1e-5,
        type=float,
        help="Learning rate for the pretrained layers",
    )
    parser.add_argument(
        "--lr_multiplier",
        default=0.9,
        type=float,
        help="Multiplier for the optimiser",
    )
    parser.add_argument(
        "--scheduler_step",
        default=5,
        type=int,
        help="scheduler_step for the optimiser",
    )
    parser.add_argument(
        "--use_scheduler",
        type=int,
        default=1,
        help="use pytorch scheduler",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.000001,
        type=float,
        help="Weight decay for the optimiser",
    )
    parser.add_argument(
        "--stop_lr",
        default=1e-5,
        type=float,
        help="Stopping learning rate for the scheduler",
    )

    # Model training parameters
    parser.add_argument(
        "--num_epochs",
        default=400,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=5,
        help="Evaluate keyframes for every 'num keyframe'",
    )

    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        help="start_epoch",
    )
    parser.add_argument(
        "--use_mask",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--num_mask",
        default=5,
        type=int,
        help="num masks for the features",
    )

    # Model parameters
    parser.add_argument(
        "--backbone",
        type=str,
        default="posecnn",
        help="vgg/resnet/posecnn/transformer",
    )
    parser.add_argument(
        "--gru",
        type=str,
        default="simple",
        help="conv, simple, beit, swin",
    )
    parser.add_argument(
        "--num_layers_backbone",
        default=4,
        type=int,
        help="number of layers in the transformer model in backbone",
    )
    parser.add_argument(
        "--num_layers_gru",
        default=4,
        type=int,
        help="number of layers in the transformer model in time series",
    )
    parser.add_argument(
        "--num_heads_gru",
        default=4,
        type=int,
        help="number of heads in the transformer model in time series",
    )
    parser.add_argument(
        "--inter_dim",
        default=768,
        type=int,
        help="number of inter dimension for the GPT2 module in the GRU",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--learn_refine",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--transform_previous",
        default=1,
        type=int,
        help="transform previous features using camera transformation",
    )
    parser.add_argument(
        "--restart_optimiser",
        default=0,
        type=int,
    )
    # Loss parameters
    parser.add_argument(
        "--losses",
        "--losses-list",
        nargs="+",
        type=str,
        default=["ADD quat"],
    )
    parser.add_argument(
        "--alpha",
        default=1,
        type=float,
        help="multiplier for depth and label losses",
    )
    parser.add_argument(
        "--beta",
        default=1,
        type=float,
        help="multiplier for ADD loss",
    )
    parser.add_argument(
        "--gamma",
        default=1,
        type=float,
        help="multiplier for translation loss",
    )
    parser.add_argument(
        "--use_depth",
        default=0,
        type=int,
        help="Use depth",
    )
    parser.add_argument(
        "--use_posecnn",
        default=1,
        type=int,
        help="Use posecnn bbox for evaluate",
    )
    parser.add_argument(
        "--predict_future",
        default=0,
        type=int,
        help="Predict future frame features like in AVT+",
    )
    # loading parameters
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="num_workers",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="val.txt",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="train.txt",
    )
    parser.add_argument(
        "--keyframe_file",
        type=str,
        default="keyframe.txt",
    )
    parser.add_argument(
        "--add_jitter",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch size for the dataset",
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--roi_noise",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_gpu",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--split",
        "--split-list",
        nargs="+",
        type=str,
        default=["train", "test"],
    )

    args = parser.parse_args()

    # Checkpoint stuff
    if not osp.isdir(args.model_dir):
        os.mkdir(args.model_dir)
    args.model_root = args.model_dir

    path = "{}_lr_{}_wd_{}".format(args.env_name, str(args.lr),
                                   str(args.weight_decay))
    args.writer_dir = os.path.join("./logs", args.env_name)

    if len(args.losses[0]) > 1:
        args.losses = str2list(args.losses[0])

    if args.restore_file is not None and args.restore_file != "None":
        unchange_var = [
            "predict_future",
            "use_label",
            "use_depth",
            "alpha",
            "beta",
            "gamma",
            "transform_previous",
            "learn_refine",
            "gru",
            "backbone",
            "num_mask",
            "use_mask",
        ]
        assert os.path.isfile(args.restore_file) is False, (
            f'Give a path relative to the model directory')

    checkpoint_dir_path = osp.join(args.model_dir, path)
    if not osp.isdir(checkpoint_dir_path):
        os.mkdir(checkpoint_dir_path)
    args.model_dir = checkpoint_dir_path

    args.hostname = os.uname().nodename

    return args
