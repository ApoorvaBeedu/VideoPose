# Video based Object Pose Estimation using Transformers

This directory contains implementation of paper [Video based Object 6D Pose Estimation using Transformers](https://arxiv.org/pdf/2210.13540.pdf).\
Accepted into [NeuRIPS 2022 Workshop on Vision Transformers: Theory and Applications](https://sites.google.com/view/vtta-neurips2022/home). 

<img src="transformer_videopose.png" width="500" style="display: block; margin: 0 auto"/>


If this code helps with your work, please cite: 
```bibtex
@article{beedu2022video,
  title={Video based Object 6D Pose Estimation using Transformers},
  author={Beedu, Apoorva and Alamri, Huda and Essa, Irfan},
  journal={arXiv preprint arXiv:2210.13540},
  year={2022}
}
```


### Environment setup
Please install all the requirements using requirements.txt
```pip3 install -r requirements.txt```

### Directory setup
Create a `./evaluation_results_video`, `wandb`, `logs`, `output` and `model` folders. 

### Arguments
Arguments and their defaults are in ```arguments.py```
-  ```backbone``` swin or beit
-  ```use_depth``` To use ground-truth depth during training
-  ```restore_file``` name of the file in --model_dir_path containing weights to reload before training
-  ```lr``` Learning rate for the optimiser
-  ```batch_size``` Batch size for the dataset
-  ```workers``` num_workers
-  ```env_name``` environment name for wandb, which is also the checkpoint name


### Setting up dataset

Download the entire YCB dataset from https://rse-lab.cs.washington.edu/projects/posecnn/    
The data folder looks like
          
```
train_eval.py
dataloader.py
├── data
│   ├── YCB
│   │   └── data
│   │       ├── 0000
│   │       └── 0001
│   │   └── models
│   │   └── train.txt
│   │   └── keyframe.txt
│   │   └── val.txt

```
### Execution

```bash
python3 train_eval.py --batch_size=8 --lr=0.0001 --backbone=swin --predict_future=1 --use_depth=1 --video_length=5 --workers=12
```
# Video based Object Pose Estimation using Transformers

This directory contains implementation for estimating 6D object poses from videos.

### Environment setup
Please install all the requirements using requirements.txt
```pip3 install -r requirements.txt```

### Directory setup
Create a `./evaluation_results_video`, `wandb`, `logs`, `output` and `model` folders. 

### Arguments
Arguments and their defaults are in ```arguments.py```
-  ```backbone``` swin or beit
-  ```use_depth``` To use ground-truth depth during training
-  ```restore_file``` name of the file in --model_dir_path containing weights to reload before training
-  ```lr``` Learning rate for the optimiser
-  ```batch_size``` Batch size for the dataset
-  ```workers``` num_workers
-  ```env_name``` environment name for wandb, which is also the checkpoint name


### Setting up dataset

Download the entire YCB dataset from https://rse-lab.cs.washington.edu/projects/posecnn/    

Download the checkpoint from https://drive.google.com/drive/folders/1lQh3G7KN-SHb7B-NYpqWj55O1WD4E9s6?usp=sharing 

Add the checkpoint to ./model/Videopose/last_checkpoint_0000.pt, and pass the argument ```--restore_file=Videopose``` during training to start from a checkpoint. If no ```start_epoch``` is mentioned, the training will restart from the last checkpoint.

The data folder looks like
          
```
train_eval.py
dataloader.py
├── data
│   ├── YCB
│   │   └── data
│   │       ├── 0000
│   │       └── 0001
│   │   └── models
│   │   └── train.txt
│   │   └── keyframe.txt
│   │   └── val.txt

```
### Execution

The project uses [wandb](https://wandb.ai) for visualisation.

```Main``` branch uses ```-posecnn.mat``` files, that I manually generated for every frame in the dataset using Posecnn repository. If you do not have those files, ```v1``` is the branch to use. 


```bash
python3 train_eval.py --batch_size=8 --lr=0.0001 --backbone=swin --predict_future=1 --use_depth=1 --video_length=5 --workers=12
```

### Evaluation
Evaluation currently runs only on one GPU. 
```bash
python3 train_eval.py --batch_size=8 --backbone=swin --predict_future=1 --use_depth=1 --video_length=5 --workers=12  --restore_file=Videopose --split=eval
```
The command will create several mat files for the keyframes and also saves images into a folder. To evaluate the mat files, please use the [YCBToolBox](https://github.com/yuxng/YCB_Video_toolbox). 
