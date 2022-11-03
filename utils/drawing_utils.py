import cv2
import numpy as np
import torch
from transforms3d.quaternions import quat2mat


class DrawFromPoints:

    def __init__(self, g_draw):
        self.g_draw = g_draw

    def drawLine(self, point, point2, lineColor, lineWidth):
        '''draws line on image'''
        if not point is None and point2 is not None:
            self.g_draw.line([point, point2], fill=lineColor, width=lineWidth)

    def drawDot(self, point, pointColor, pointRadius):
        '''draws dot (filled circle) on image'''
        if point is not None:
            xy = [
                point[0] - pointRadius, point[1] - pointRadius,
                point[0] + pointRadius, point[1] + pointRadius
            ]
            self.g_draw.ellipse(xy, fill=pointColor, outline=pointColor)

    def drawCube(self, points, color=(255, 0, 0)):
        '''
        draws cube with a thick solid line across
        the front top edge and an X on the top face.
        '''

        lineWidthFordrawing = 2

        # draw front
        self.drawLine(points[0], points[1], color, lineWidthFordrawing)
        self.drawLine(points[1], points[2], color, lineWidthFordrawing)
        self.drawLine(points[3], points[2], color, lineWidthFordrawing)
        self.drawLine(points[3], points[0], color, lineWidthFordrawing)

        # draw back
        self.drawLine(points[4], points[5], color, lineWidthFordrawing)
        self.drawLine(points[6], points[5], color, lineWidthFordrawing)
        self.drawLine(points[6], points[7], color, lineWidthFordrawing)
        self.drawLine(points[4], points[7], color, lineWidthFordrawing)

        # draw sides
        self.drawLine(points[0], points[4], color, lineWidthFordrawing)
        self.drawLine(points[7], points[3], color, lineWidthFordrawing)
        self.drawLine(points[5], points[1], color, lineWidthFordrawing)
        self.drawLine(points[2], points[6], color, lineWidthFordrawing)

        # draw dots
        self.drawDot(points[0], pointColor=color, pointRadius=4)
        self.drawDot(points[1], pointColor=color, pointRadius=4)

        # draw x on the top
        self.drawLine(points[0], points[5], color, lineWidthFordrawing)
        self.drawLine(points[1], points[4], color, lineWidthFordrawing)


class DrawFromPose:

    def __init__(self):
        self.class_colors = [(255, 255, 255), (255, 0, 0), (0, 255, 0),
                             (0, 0, 255), (255, 255, 0), (255, 0, 255),
                             (0, 255, 255), (128, 0, 0), (0, 128, 0),
                             (0, 0, 128), (128, 128, 0), (128, 0, 128),
                             (0, 128, 128), (64, 0, 0), (0, 64, 0), (0, 0, 64),
                             (64, 64, 0), (64, 0, 64), (0, 64, 64),
                             (192, 0, 0), (0, 192, 0), (0, 0, 192)]

    def drawModel(self,
                  img,
                  poses_gt=None,
                  poses_est=None,
                  bbox=None,
                  intrinsic_matrix=None,
                  points=None,
                  cls_indexes=None,
                  classes=None,
                  alpha=0.6):
        imm = (img.copy() * 255).transpose(1, 2, 0)
        im_o = imm.copy()
        im = imm.copy()
        im1 = imm.copy()
        im2 = imm.copy()
        loss = 0
        loss_str = ""
        # Iterating over different boxes in one image
        for i in range(0, poses_est.size(0)):
            cls = int(cls_indexes[i])
            # To maintain same number of boxes, data is appendded with -1. See dataloader.py collate_fn
            if cls > 0:
                # extract 3D points
                x3d = np.ones((4, points.shape[1]), dtype=np.float32)
                x3d[0, :] = points[cls - 1, :, 0]
                x3d[1, :] = points[cls - 1, :, 1]
                x3d[2, :] = points[cls - 1, :, 2]

                # projection
                RT_est = np.zeros((3, 4), dtype=np.float32)
                RT_est[:3, :3] = quat2mat(poses_est[i][:4].cpu().numpy())
                RT_est[:, 3] = poses_est[i][4:7].cpu().numpy()

                RT = poses_gt[i].cpu().numpy()

                loss = np.abs(
                    np.subtract(np.matmul(RT_est, x3d),
                                np.matmul(RT, x3d))).sum(axis=0).mean()
                loss_l2 = np.sqrt(
                    np.power(
                        np.subtract(np.matmul(RT_est, x3d), np.matmul(RT,
                                                                      x3d)),
                        2).sum(axis=0)).mean()
                loss_str = loss_str + "{}_l1_{:.3f}_l2_{:.3f} ".format(
                    cls, loss, loss_l2)
                x2d = np.matmul(intrinsic_matrix, np.matmul(RT_est, x3d))
                x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
                x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])
                x2d_gt = np.matmul(intrinsic_matrix, np.matmul(RT, x3d))
                x2d_gt[0, :] = np.divide(x2d_gt[0, :], x2d_gt[2, :])
                x2d_gt[1, :] = np.divide(x2d_gt[1, :], x2d_gt[2, :])

                for poi in range(0, x2d.shape[1]):
                    try:
                        y, x = int(x2d[1, poi]), int(x2d[0, poi])
                        cv2.circle(im, (x, y), 1, self.class_colors[cls - 1])
                    except:
                        pass
                image = cv2.addWeighted(im, alpha, im_o, 1 - alpha, 0)
                cv2.putText(image, 'VideoPose', (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

                for poi in range(0, x2d_gt.shape[1]):
                    try:
                        y, x = int(x2d_gt[1, poi]), int(x2d_gt[0, poi])
                        cv2.circle(im1, (x, y), 1, self.class_colors[cls - 1])
                    except:
                        pass
                image2 = cv2.addWeighted(im1, alpha, im_o, 1 - alpha, 0)
                cv2.putText(image2, 'Ground Truth', (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                            cv2.LINE_AA)

        # for q in range(bbox.size(0)):
        #     cls = int(cls_indexes[q])
        #     box = [int(b.cpu().numpy()) for b in bbox[q, :]]
        #     cv2.rectangle(im1, (box[0], box[1]), (box[2], box[3]),
        #                   color=self.class_colors[cls - 1])

        # cv2.rectangle(im, (5, 5), (460, 30), (0, 0, 0), -1)
        # cv2.putText(im, loss_str, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (255, 255, 255), 1, cv2.LINE_AA)
        return np.hstack((image, image2)).astype(np.uint8).transpose(2, 0, 1)


class VisualiseSegmentation:

    def __init__(self):
        self.class_colors = np.asarray([(0, 0, 0), (255, 255, 255),
                                        (255, 0, 0), (0, 255, 0), (0, 0, 255),
                                        (255, 255, 0), (255, 0, 255),
                                        (0, 255, 255),
                                        (128, 0, 0), (0, 128, 0), (0, 0, 128),
                                        (128, 128, 0), (128, 0, 128),
                                        (0, 128, 128), (64, 0, 0), (0, 64, 0),
                                        (0, 0, 64), (64, 64, 0), (64, 0, 64),
                                        (0, 64, 64), (192, 0, 0), (0, 192, 0),
                                        (0, 0, 192)])

    def display_label(self, features):
        temp = features
        if len(features.shape) > 2:
            temp = np.argmax(features, axis=0)
        img = self.class_colors[temp.astype(int)]
        return img

    def iou(self, label_est, label_gt):
        union = ((label_est + label_gt) > 0).sum()
        intersection = (label_est * label_gt).sum()
        iou = intersection / union
        return iou

    def draw(self, x, label_gt, label_est):
        temp_labels_gt = label_gt[-1].detach().cpu().numpy().transpose(1, 2, 0)
        if label_est is not None:
            # Displaying ious
            ious = self.iou(label_est, label_gt).detach().cpu().numpy()
            iou_string = "iou: " + "{0}".format(ious)
            temp_label = label_est[-1].detach().cpu().numpy().transpose(
                1, 2, 0)
            img = np.hstack((temp_label, temp_labels_gt)).astype(np.uint8)
        else:
            img = np.hstack((temp_labels_gt, temp_labels_gt)).astype(np.uint8)
        return torch.from_numpy(img.transpose(2, 0, 1)).type_as(x)


class VisualiseDepth:

    def __init__(self):
        pass

    @staticmethod
    def display_depth(depth, bits=1):
        """Write depth map to pfm and png file.

        Args:
            path (str): filepath without extension
            depth (array): depth
        """
        depth_min = depth.min()
        depth_max = depth.max()

        max_val = (2**(8 * bits)) - 1

        if depth_max - depth_min > np.finfo("float").eps:
            out = max_val * (depth - depth_min) / (depth_max - depth_min)
        else:
            out = 0
        return out / out.max()

    def draw(self, x, depth_est, depth_gt):
        # Visualising only the last batch image
        if depth_gt[-1].max() != 0:
            depth_gt[-1] = depth_gt[-1] / depth_gt[-1].max()
        if depth_est[-1].max() != 0:
            depth_est[-1] = depth_est[-1] / depth_est[-1].max()

        diff = torch.clamp((depth_gt[-1] - depth_est[-1]), -2, 2).abs().float()
        temp_depth = ((diff.type_as(x)) *
                      125.0).squeeze(0).detach().cpu().numpy()

        img = np.hstack(
            (np.tile(temp_depth, (3, 1, 1)).transpose(1, 2, 0),
             np.tile((depth_est[-1] * 125.0).squeeze(0).detach().cpu().numpy(),
                     (3, 1, 1)).transpose(1, 2, 0),
             np.tile((depth_gt[-1] * 125.0).squeeze(0).detach().cpu().numpy(),
                     (3, 1, 1)).transpose(1, 2, 0))).transpose(2, 0, 1)

        return torch.from_numpy(img.astype(np.uint8)).type_as(x)


class VisualiseFlow:

    def __init__(self):
        pass

    def flow2img(self, flow_data):
        """
        convert optical flow into color image
        :param flow_data:
        :return: color image
        """
        # print(flow_data.shape)
        # print(type(flow_data))
        u = flow_data[:, :, 0]
        v = flow_data[:, :, 1]

        UNKNOW_FLOW_THRESHOLD = 1e7
        pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
        pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
        idx_unknown = (pr1 | pr2)
        u[idx_unknown] = v[idx_unknown] = 0

        # get max value in each direction
        maxu = -999.
        maxv = -999.
        minu = 999.
        minv = 999.
        maxu = max(maxu, np.max(u))
        maxv = max(maxv, np.max(v))
        minu = min(minu, np.min(u))
        minv = min(minv, np.min(v))

        rad = np.sqrt(u**2 + v**2)
        maxrad = max(-1, np.max(rad))
        u = u / maxrad + np.finfo(float).eps
        v = v / maxrad + np.finfo(float).eps

        img = self.compute_color(u, v)

        idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
        img[idx] = 0

        return np.uint8(img)

    def compute_color(self, u, v):
        """
        compute optical flow color map
        :param u: horizontal optical flow
        :param v: vertical optical flow
        :return:
        """

        height, width = u.shape
        img = np.zeros((height, width, 3))

        NAN_idx = np.isnan(u) | np.isnan(v)
        u[NAN_idx] = v[NAN_idx] = 0

        colorwheel = self.make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2 + v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

        return img

    def make_color_wheel(self):
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY,
                   1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col:col + YG, 0] = 255 - \
            np.transpose(np.floor(255 * np.arange(0, YG) / YG))
        colorwheel[col:col + YG, 1] = 255
        col += YG

        # GC
        colorwheel[col:col + GC, 1] = 255
        colorwheel[col:col + GC,
                   2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
        col += GC

        # CB
        colorwheel[col:col + CB, 1] = 255 - \
            np.transpose(np.floor(255 * np.arange(0, CB) / CB))
        colorwheel[col:col + CB, 2] = 255
        col += CB

        # BM
        colorwheel[col:col + BM, 2] = 255
        colorwheel[col:col + BM,
                   0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
        col += +BM

        # MR
        colorwheel[col:col + MR, 2] = 255 - \
            np.transpose(np.floor(255 * np.arange(0, MR) / MR))
        colorwheel[col:col + MR, 0] = 255

        return colorwheel

    def draw(self, x, flow_gt, flow_est):
        flow_gt = flow_gt[-1].detach().cpu().numpy().transpose(1, 2, 0)
        img_gt = self.flow2img(flow_gt)
        if flow_est is not None:
            img_est = self.flow2img(
                flow_est[-1].detach().cpu().numpy().transpose(1, 2, 0))
            img = np.hstack((img_est, img_gt)).astype(np.uint8)
        else:
            img = np.hstack((img_gt, img_gt)).astype(np.uint8)
        return torch.from_numpy(img.transpose(2, 0, 1)).type_as(x)

