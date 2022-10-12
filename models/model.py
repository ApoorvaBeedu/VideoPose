import copy
import os
import time
from typing import Dict

import dataloader
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torchvision import models as torch_model
from torchvision import ops as ops
from transformers import BeitModel, SwinModel

from models.enums import Output
from utils.drawing_utils import VisualiseDepth, VisualiseFlow

from . import loss
from .convgru import ConvGRU, Transformer
from .utils import get_updated_pos, weight_initialise

dir_path = os.path.dirname(os.path.realpath(__file__))
root = dir_path + "/data/LOV"

# _, points_dense = dataloader.load_object_points_dense(root)
# points_dense = torch.from_numpy(points_dense)

# ================================ Models ================================


class PoseCNN(nn.Module):
    def __init__(self, args, input_channels=3):
        super().__init__()

        self.args = args

        self.conv1_1 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][0:2])
        self.conv1_2 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][2:4])
        self.pool1 = nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  padding=0,
                                  dilation=1,
                                  ceil_mode=False)
        self.conv2_1 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][5:7])
        self.conv2_2 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][7:9])
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  padding=0,
                                  dilation=1,
                                  ceil_mode=False)
        self.conv3_1 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][10:12])
        self.conv3_2 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][12:14])
        self.conv3_3 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][14:16])
        self.pool3 = nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  padding=0,
                                  dilation=1,
                                  ceil_mode=False)
        self.conv4_1 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][17:19])
        self.conv4_2 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][19:21])
        self.conv4_3 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][21:23])
        self.pool4 = nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  padding=0,
                                  dilation=1,
                                  ceil_mode=False)
        self.conv5_1 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][24:26])
        self.conv5_2 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][26:28])
        self.conv5_3 = nn.Sequential(
            *list(torch_model.vgg16(pretrained=False).children())[0][28:30])

        self.relu = nn.ReLU()

        self.apply(weight_initialise)
        self.mean_ = torch.DoubleTensor([122.7717, 115.9465, 102.9801])  # RGB
        self.mean = torch.DoubleTensor([0.485, 0.456, 0.406])
        self.std = 1 / torch.DoubleTensor([0.229, 0.224, 0.225])

    def unfreeze_layers(self):
        layers = [self.conv5_3.parameters(), self.conv5_2.parameters()]
        for i in layers:
            for param in i:
                param.requires_grad = True

    def convert_image(self, x):
        batch = x.shape[0]
        y = x.div((self.std).repeat(batch, 1).view(
            batch, -1, 1, 1).type_as(x)) + self.mean.repeat(batch, 1).view(
                batch, -1, 1, 1).type_as(x)
        return (y * 255) - self.mean_.repeat(batch, 1).view(batch, -1, 1,
                                                            1).type_as(x)

    def forward(self, x):
        x = self.convert_image(x)[:, [2, 1, 0], :, :]
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)
        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        conv3_3 = self.conv3_3(conv3_2)
        pool3 = self.pool3(conv3_3)
        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        conv4_3 = self.conv4_3(conv4_2)
        pool4 = self.pool4(conv4_3)
        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)
        conv5_3 = self.conv5_3(conv5_2)
        pool_score = F.interpolate(conv4_3, conv5_3.shape[2:]) + conv5_3

        return pool_score, conv4_3


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,
                               out_channels=256,
                               kernel_size=3,
                               padding=1),
            nn.GroupNorm(8, 256, eps=1e-05),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=3,
                               padding=1),
            nn.GroupNorm(8, 128, eps=1e-05),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1))

        self.softmax = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_softmax=False, use_sigmoid=False):
        x = self.conv1(x)
        x = self.conv2(x)
        out = self.conv3(x)

        if use_sigmoid:
            out = self.sigmoid(out)
        if use_softmax:
            out = self.softmax(out)

        return out, x


class Linear(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels
        self.fcn1 = nn.Linear(in_channels, 512)
        self.fcn_r = nn.Linear(512, 3)
        self.fcn_xy = nn.Linear(512, 2)
        self.fcn_z = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, features):
        x = features.view(features.size(0), self.in_channels)
        x = self.dropout(self.relu(self.fcn1(x)))

        output_r_xyz = self.fcn_r(x)
        output_w = (1 - output_r_xyz.norm(dim=1)).reshape(-1, 1)
        output_r = torch.cat((output_w, output_r_xyz), dim=1)
        output_xy = self.fcn_xy(x)
        output_z = self.fcn_z(x)
        output_t_est = (torch.cat((output_xy, output_z),
                                  dim=1).type_as(x).requires_grad_())

        return output_r, output_t_est


class VideoPose(nn.Module):
    def __init__(self, arg):
        super().__init__()

        self.args = arg
        self.h2 = 16

        # Backbone feature extraction stuff
        if self.args.use_depth:
            self.args.inter_dim *= 2
        self.gru_in_channel = self.args.inter_dim
        if self.args.backbone == "posecnn":
            self.backbone = PoseCNN(arg)
            self.gru_in_channel = 512  # Image_feat + depth
            if self.args.use_depth:
                self.gru_in_channel += 128
            if self.args.use_depth:
                self.depth_decoder = Decoder(512, 1)
                weight_initialise(self.depth_decoder)
        elif self.args.backbone == "swin":
            self.backbone = SwinModel.from_pretrained(
                "microsoft/swin-tiny-patch4-window7-224")
            if self.args.use_depth:
                self.linear_depth = nn.Linear(768 * 2, 768)
        elif self.args.backbone == "beit":
            self.backbone = BeitModel.from_pretrained(
                "microsoft/beit-base-patch16-224-pt22k-ft22k")
            self.deleteEncodingLayers(num_layers=self.args.num_layers_backbone)
            self.backbone.config.num_hidden_layers = self.args.num_layers_backbone
            if self.args.use_depth:
                self.linear_depth = nn.Linear(768 * 2, 768)

        self.nc_memory = 128
        self.nc_fused = 256
        self.pool_size = 12
        self.regressor_in = self.nc_fused * (self.pool_size**2)

        if self.args.backbone == "posecnn":
            self.gru_in_channel *= (self.pool_size**2)
        self.gru = Transformer(self.gru_in_channel, self.args)

        # Future prediction stuff
        self.regressor_in = self.args.inter_dim
        self.future_predictor = nn.Sequential(
            nn.Linear(self.regressor_in, self.regressor_in),
            nn.ReLU(inplace=True),
            nn.Linear(self.regressor_in, self.regressor_in))

        # Pose prediction Linear MLP stack
        self.regressor = Linear(self.regressor_in)

        # Loss stuff
        self.loss_fn = loss.Loss(self.args.losses, dataloader, arg)
        self.loss_dict = None

        # General layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

        # For depth and segmentation visualisation
        self.depth_visualisation = VisualiseDepth()
        self.flow_visualisation = VisualiseFlow()

    def deleteEncodingLayers(self,
                             num_layers):  # must pass in the full bert model
        oldModuleList = self.backbone.encoder.layer
        newModuleList = nn.ModuleList()

        # Now iterate over all layers, only keepign only the relevant layers.
        for i in range(0, (num_layers)):
            newModuleList.append(oldModuleList[i])

        # create a copy of the model, modify it with the new list, and return
        self.backbone = copy.deepcopy(self.backbone)
        self.backbone.encoder.layer = newModuleList

    def get_params(self, pretrained_lr):
        params = [
            {
                "params": self.gru.parameters(),
                "lr": self.args.lr
            },
            {
                "params": self.regressor.parameters(),
                "lr": self.args.lr
            },
            {
                "params": self.backbone.parameters(),
                "lr": pretrained_lr
            },
        ]
        if self.args.use_depth and self.args.backbone == "posecnn":
            params.append({
                "params": self.depth_decoder.parameters(),
                "lr": self.args.lr
            })
        return params

    def format_batch(self, batch):
        # Converts a list of tensors to one tensor, bind basically
        for keys, val in batch.items():
            if isinstance(val[0], torch.Tensor):
                batch[keys] = torch.stack(val)
        return batch

    def get_roi_features(self, batch):
        bbox = batch["bbox"]
        x = batch["image"]
        y = batch["depth"]
        _b, _t, _c, _h, _w = x.shape
        _n = bbox.shape[2]
        x = x.reshape(_b * _t, _c, _h, _w)
        y = y.reshape(_b * _t, _h, _w)
        bbox = bbox.reshape(-1, _n, 4)
        ind = (torch.arange(bbox.shape[0]).unsqueeze(1).repeat(
            1, bbox.size(1)).reshape(-1, 1))
        ind = ind.type_as(bbox)
        bbox_roi = torch.cat((ind, bbox.reshape(-1, 4).float()), dim=1).float()
        batch["bbox_roi"] = bbox_roi
        batch["depth_pred"] = None
        if self.args.backbone == "beit" or self.args.backbone == "swin":
            output_size = (224, 224)
            roi_images = ops.roi_align(x.reshape(-1, 3, _h, _w), bbox_roi,
                                       output_size)
            # Get features from transformer backbone
            t0 = time.time_ns()
            feats = self.backbone(roi_images)
            # Need to add a temporal dimension
            feats = feats["pooler_output"].unsqueeze(1)
            t1 = time.time_ns()
            if self.args.use_depth:
                output_size = (224, 224)
                roi_depths = ops.roi_align(
                    y.repeat(3, 1, 1).reshape(-1, 3, _h, _w), bbox_roi,
                    output_size)
                depth_feats = self.backbone(roi_depths)
                depth_feats = depth_feats["pooler_output"].unsqueeze(1)
                feats = torch.cat((feats, depth_feats), axis=2)
                # feats = self.linear_depth(self.relu(feats))
            feats = feats.reshape(_b, _t, _n, -1, self.args.inter_dim)

        elif self.args.backbone == "posecnn":
            output_size = (12, 12)
            t0 = time.time_ns()
            out_x, _ = self.backbone(x)
            ratio = out_x.shape[2] / x.shape[2]
            y_local = ops.roi_align(
                out_x,
                bbox_roi,
                output_size=output_size,
                spatial_scale=ratio,
            )
            y_depth = torch.tensor(()).type_as(x)
            if self.args.use_depth:
                depth_pred, penultimate_depth = self.depth_decoder(out_x)
                ratio_depth = penultimate_depth.shape[2] / x.shape[2]
                batch["depth_pred"] = depth_pred
                y_depth = ops.roi_align(
                    penultimate_depth,
                    bbox_roi,
                    output_size=output_size,
                    spatial_scale=ratio_depth,
                )

            # Roi pooling
            loc = torch.randint(0, self.pool_size - 5, (self.args.num_mask, ))
            feats = torch.cat((y_local, y_depth),
                              dim=1).type_as(x).requires_grad_()
            if self.backbone.training:
                if self.args.use_mask:
                    with torch.no_grad():
                        for l_ in loc:
                            feats[:, :, l_:l_ + 5,
                                  l_:l_ + 5] = torch.Tensor([0])
            t1 = time.time_ns()
            feats = feats.reshape(_b, _t, _n, -1, output_size[0],
                                  output_size[1])
        logger.info(f"Time taken by the backbone {(t1-t0)/10**9} seconds")
        return feats

    def temporal_block(self, feats, batch):
        feat = self.gru(feats)
        return feat

    def get_pose(self, feats, batch):
        bbox_roi = batch["bbox_roi"]
        _b, _t, _n, _ = batch["bbox"].shape
        if len(feats.shape) > 2:
            feats = feats.reshape(-1, self.args.inter_dim)
        output_r, output_t_est = self.regressor(feats)

        # Feats, thus, output is _bx_nx_t while bbox is _bx_tx_n
        bbox_roi = bbox_roi.reshape(_b, _t, _n, 5).permute(0, 2, 1,
                                                           3).reshape(-1, 5)

        if self.args.learn_refine:
            # We are assuming that the output of the model is the delta of the center from the ROI.
            output_t = get_updated_pos(bbox_roi, output_t_est,
                                       batch["intrinsic"])
        else:
            output_t = output_t_est

        output_r = output_r.view(_b, _n, _t, 4).permute(0, 2, 1, 3)
        output_t = output_t.view(_b, _n, _t, 3).permute(0, 2, 1, 3)
        logger.info("output_t_est {0}".format(output_t_est[-1]))
        return output_r, output_t

    def forward(
        self,
        batch,
        rank,
    ):
        # Adding indices for the ROI pooling part

        x = batch["image"]
        _b, _t, _, _, _ = x.shape
        _n = batch["bbox"].shape[2]  # b x t x n x 4

        # Get ROI features
        t0 = time.time()
        backbone_features = self.get_roi_features(batch)
        t1 = time.time()
        logger.info(f"Time taken by backbone {(t1-t0)} sec")

        # Get temporal features
        temporal_features = self.temporal_block(backbone_features, batch)
        t1 = time.time()
        logger.info(f"Time taken by temporal {(t1-t0)} sec")
        # temporal features are of shape _b*_n, _t, 768
        temporal_features = temporal_features.reshape(-1, _t,
                                                      self.args.inter_dim)

        t1 = time.time()
        logger.info(f"Time taken by one forward pass {(t1-t0)} sec")
        output_r, output_t = self.get_pose(temporal_features, batch)
        t1 = time.time()
        logger.info(f"Time taken by one forward pass {(t1-t0)} sec")

        # Getting losses for the predictions
        future_feat = self.future_predictor(temporal_features)
        loss_, loss_dict = self.loss_fn(
            x,
            output_r,
            output_t,
            temporal_features[:, 1:, :],
            future_feat[:, :_t - 1, :],
            batch["poses"],
            batch["extrinsic"],
            batch["intrinsic"],
            batch["cls_indices"],
            self.args.losses,
        )

        output: Output = {
            "_R": output_r,
            "_T": output_t,
            "output_losses": loss_dict,
            "loss_value": loss_,
        }

        return output
