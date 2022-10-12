import json
import os
import pdb
import random
import time

import numpy as np
import scipy.io as sio
import torch
# To read stuff
from bbox import BBox2D
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from transforms3d.quaternions import quat2mat

from utils.pytorch_utils import ColorJitter


def collate_fn(batch):
    # batch_new = []
    # datum = {}
    for b in batch:
        for k, v in b.items():
            b[k] = torch.stack(v)
        # batch_new.append(datum)

    # get sequence lengths
    # max_length = max([t["bbox"].shape[1] for t in batch])
    max_length = 9
    # padd
    for data in batch:
        for i in range(0, max_length - data["bbox"].shape[1]):
            b, c, h, w = data["image"].shape
            dummy = torch.tensor([[0, 0, h, w]]).repeat(b, 1, 1)
            data["bbox"] = torch.cat((data["bbox"], dummy), dim=1)
            data["posecnn_bbox"] = torch.cat((data["posecnn_bbox"], dummy),
                                             dim=1)
            dummy = torch.zeros((b, 1, 3, 4), dtype=float)
            data["poses"] = torch.cat((data["poses"], dummy), dim=1)
            data["posecnn_poses"] = torch.cat((data["posecnn_poses"], dummy),
                                              dim=1)
            dummy = torch.tensor([[-1]]).repeat(b, 1)
            data["cls_indices"] = torch.cat((data["cls_indices"], dummy),
                                            dim=1)
    # Converts list of batch to tensor
    batch = default_collate(batch)
    return batch


class VideoLoader(Dataset):

    def __init__(
        self,
        data_root_path,
        train_file_path,
        transform=None,
        add_jitter=False,
        add_noise=False,
        video_length=20,
        add_rot=False,
        add_translation=False,
        roi_noise=1,
        single=False,
        step=0,
        is_train=False,
    ):
        super(VideoLoader, self).__init__()

        self.data_root_path = data_root_path
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.data_dir = os.path.join(data_root_path, "data")

        self.transform = transform
        self.jitter_transform = ColorJitter()
        self.add_noise = add_noise
        self.add_rot = add_rot
        self.add_translation = add_translation
        self.add_jitter = add_jitter
        self.roi_noise = roi_noise
        self.video_length = video_length
        self.single = single
        self.step = step
        self.is_train = is_train

        self.augmentation_data = []  # angle, cx, cy
        self.gaussian_noise = None

        self.classes = get_classes(self.data_root_path)
        self.common_objects = []

        self.train_list = self.get_image_data_list(train_file_path)
        self.num_samples = len(self.train_list)
        self.last_fn_dict = self.get_last_filename()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        This function, given the index, gets video sequence containing 20
        images. Performs data augmentation (add gaussian noise, rot and
        translation) randomly
        """
        data = {
            "image": [],
            "poses": [],
            "cls_indices": [],
            "extrinsic": [],
            "intrinsic": [],
            "file_indices": [],
            "bbox": [],
            "label": [],
            "depth": [],
            "is_keyframe": [],
            "posecnn_bbox": [],
            "posecnn_poses": [],
        }

        if self.is_train:
            self.step = np.random.randint(1, 10, 1)[0]
        flist = self.train_list[index]
        if not self.single:
            flist = self.get_video_list(flist, self.step)
        if not isinstance(flist, list):
            flist = [flist]
        bbox_data = self.load_bbox_files(flist)
        self.get_common_objects(flist, bbox_data)

        self.augmentation_data = []
        if np.random.random_sample([1]) > 0.5:
            # Angle and translation are randomly sampled from [-20, 20)
            rand_angle = 40 * np.random.random_sample((1)) - 20
            rand_c = 40 * np.random.random_sample((2)) - 20
            self.augmentation_data = [rand_angle, rand_c]

        height, width = 480, 640
        noise = np.zeros((height, width))
        if np.random.random_sample([1]) > 0.5:
            # Noise is added before normalisation
            noise = np.random.randn(height, width) * 0.1
        self.gaussian_noise = np.tile(noise, (3, 1, 1)).transpose(1, 2, 0)
        self.jitter_transform.set_factors()

        for ind, fl in enumerate(flist):
            fl_list = fl.split("/")
            datum = self.get_data_post_processed(fl, bbox_data)
            (
                img,
                label,
                depth,
                poses,
                cls_indices,
                bbox,
                K,
                extrinsic,
                bbox_p,
                poses_p,
            ) = datum
            is_keyframe = torch.tensor(
                fl.strip() == self.train_list[index].strip())

            data["image"].append(img)
            data["label"].append(label)
            data["depth"].append(depth)
            data["poses"].append(poses)
            data["posecnn_poses"].append(poses_p)
            data["cls_indices"].append(cls_indices)
            data["bbox"].append(bbox)
            data["posecnn_bbox"].append(bbox_p)
            data["file_indices"].append(torch.tensor([int(i)
                                                      for i in fl_list]))
            data["intrinsic"].append(K)
            data["extrinsic"].append(extrinsic)
            data["is_keyframe"].append((is_keyframe))
        return data

    @staticmethod
    def get_image_data_list(file_path):
        lis = []
        with open(file_path, "r") as file:
            lines = file.readlines()
        for i in range(0, len(lines)):
            lis.append(lines[i].strip())
        return lis

    def get_last_filename(self):
        dirs = os.listdir(self.data_dir)
        dirs.sort()
        last_filename_dict = {}
        for i in dirs:
            dir_path = os.path.join(self.data_dir, i)
            if os.path.isdir(dir_path):
                files = os.listdir(dir_path)
                files.sort()
                for file in files:
                    if file.endswith("-color.png"):
                        last_filename_dict[i] = file.replace("-color.png", "")
        return last_filename_dict

    def get_video_list(self, filename, step=None):
        vid, file = filename.strip().split("/")
        last_file = int(file)
        first_file = np.max((
            1,
            last_file - (self.video_length) * step,
        ))
        range_file = list(range(first_file, last_file + 1, step))
        if len(range_file) < self.video_length:
            for i in range(self.video_length - len(range_file)):
                range_file.insert(0, range_file[0])
        if len(range_file) >= self.video_length:
            range_file = range_file[-self.video_length:]
        file_list = ["{0}/{1}".format(vid, "%06d" % i) for i in range_file]
        return file_list

    def load_bbox_files(self, flist):
        """
        Loads bbox data for all the images in the video sequence.
        Also, randomly shuffles, to allow model to learn all objects
        Dataloader tends to crop the size of the list of unequal lengths to
        the smallest one (our case, most often 3)
        :param flist: List of filenames in a video sequence
        :return: bbox dictionary
        """
        random_order = None
        bbox_data = {}

        for fl in flist:
            dir_path = os.path.join(self.data_dir, fl)
            bbox_path = os.path.join(dir_path + "-box.txt")
            with open(bbox_path, "r") as file:
                lines = file.readlines()
            if random_order is None:
                random_order = random.sample(range(0, len(lines)), len(lines))
            bbox_data[fl] = [lines[i].strip() for i in random_order]
        return bbox_data

    def get_common_objects(self, flist, bbox_data):
        """
        Finds common cls_indices in all the 20 images in the video sequence
        :param flist: List of filenames in a video sequence
        :return: self.common_objects is stored with the common_objects
        """
        common_objects = []
        if bbox_data is None:
            self.common_objects = common_objects
            return

        for fl in flist:
            bbox_datum = bbox_data[fl]

            cls_indices = []
            for i in range(0, len(bbox_datum)):
                lis = str(bbox_datum[i]).split(" ")
                box = BBox2D([float(i) for i in lis[1::]], mode=1)
                # The h and w calculation takes care of -ve signs.
                # 5 pixel tolerance so that the bbox can be jittered.
                if ((box.x1 and box.x2) <= 640 and (box.y1 and box.y2) <= 480
                        and box.w > 2 and box.h > 2):
                    cls_indices.append(self.classes[lis[0]])
            common_objects.append(set(cls_indices))

        try:
            self.common_objects = sorted(
                list(set.intersection(*common_objects)))
        except:
            print(flist)
            pdb.set_trace()

    def get_data_post_processed(self, fl, bbox_data):
        """
        Gets the post processed data, which includes finding objects in every
        image that is a valid good object, transform all data and data
        augmentation
        :param fl:
        :return: im, label, depth, poses, cls_ind, bbox, crops, intrinsics, RT
        """
        bbox = []
        posecnn_bbox = []
        poses = []
        posecnn_poses = []
        cls_indices = []
        dir_path = os.path.join(self.data_dir, fl)

        meta_path = os.path.join(dir_path + "-meta.mat")
        posecnn_meta_path = os.path.join(dir_path + "-posecnn.mat")
        meta_data = sio.loadmat(meta_path)
        posecnn_meta_data = sio.loadmat(posecnn_meta_path)
        bbox_datum = bbox_data[fl]
        factor_depth = meta_data["factor_depth"]
        intrinsics = torch.Tensor(meta_data["intrinsic_matrix"])
        extrinsic = meta_data["rotation_translation_matrix"]
        extrinsic_hom = torch.eye(4)
        extrinsic_hom[0:3, :] = torch.from_numpy(extrinsic)

        bbox_posecnn_datum = posecnn_meta_data["rois"][:, 2:6]
        cls_posecnn_datum = posecnn_meta_data["rois"][:, 1]
        poses_posecnn_datum = posecnn_meta_data["poses"]

        # Loading all image data (rgb, segmentation and depth)
        img_path = os.path.join(dir_path + "-color.png")
        label_path = os.path.join(dir_path + "-label.png")
        depth_path = os.path.join(dir_path + "-depth.png")
        img = Image.open(img_path)
        label = np.array(Image.open(label_path))
        depth = np.array(Image.open(depth_path)) / factor_depth

        # Loading only data for object present in common_objects
        for i in range(0, len(bbox_datum)):
            lis = str(bbox_datum[i]).split(" ")
            box = BBox2D([float(i) for i in lis[1::]], mode=1)
            if self.classes[lis[0]] in self.common_objects:
                delta_m = (np.minimum(box.w, box.h) * 0.1).round()
                delta = np.random.randint(-delta_m, delta_m + 1, (4))
                bbox.append([box.x1, box.y1, box.x2, box.y2])
                cls_indices.append(self.classes[lis[0]])
                ind = list(meta_data["cls_indexes"]).index(
                    self.classes[lis[0]])
                poses.append(meta_data["poses"][:, :, ind])

                if self.roi_noise > 1:
                    posecnn_bbox.append([
                        box.x1 + delta[0],
                        box.y1 + delta[1],
                        box.x2 + delta[2],
                        box.y2 + delta[3],
                    ])
                    posecnn_poses.append(np.zeros((1, 7), dtype=float))
                else:
                    if self.classes[lis[0]] in cls_posecnn_datum:
                        (ind, ) = np.where(
                            cls_posecnn_datum == self.classes[lis[0]])
                        posecnn_bbox.append(bbox_posecnn_datum[ind, :][0])
                        posecnn_poses.append(
                            poses_posecnn_datum[ind].astype(float))
                    else:
                        # Toolbox wouldn't consider this object anyway
                        posecnn_bbox.append([box.x1, box.y1, box.x2, box.y2])
                        posecnn_poses.append(np.zeros((1, 7), dtype=float))

        datum = self.post_process_images_bbox_poses(img, depth, label, bbox,
                                                    poses, intrinsics,
                                                    posecnn_bbox,
                                                    posecnn_poses)
        im, depth, label, bbox, poses, rt_bbox_posecnn, rt_poses_posecnn = datum

        return (
            im,
            label,
            depth,
            poses,
            torch.Tensor(cls_indices),
            bbox,
            intrinsics,
            extrinsic_hom,
            rt_bbox_posecnn,
            rt_poses_posecnn,
        )

    def post_process_images_bbox_poses(self,
                                       img,
                                       depth,
                                       label,
                                       bbox,
                                       poses,
                                       K,
                                       posecnn_bbox=None,
                                       posecnn_poses=None):
        """
        Adds gaussian noise to the image, rotates and translated all other data
        :param img: Image to be transformed
        :param depth: Depth image to be transformed
        :param label: Segmentation image to be transformed
        :param bbox: Ground truth bounding box
        :param poses: Ground truth poses
        :param K: Intrinsic matrix
        :return: rt_img, rt_depth, rt_label, rt_bbox, rt_poses
        """
        if self.add_noise:
            img = img + self.gaussian_noise
            img = Image.fromarray(np.uint8(img))

        fx, fy = K[0, 0], K[1, 1]
        angle, cx, cy = 0, 0, 0
        if len(self.augmentation_data) > 0:
            if self.add_rot:
                angle = self.augmentation_data[0]
            if self.add_translation:
                cx, cy = self.augmentation_data[1]

        depth = Image.fromarray(depth)
        label = Image.fromarray(label)
        rt_img = rotate_translate_image(img,
                                        theta=angle,
                                        cx=cx,
                                        cy=cy,
                                        fillcolor=None)
        rt_depth = rotate_translate_image(depth, theta=angle, cx=cx, cy=cy)
        rt_label = rotate_translate_image(label, theta=angle, cx=cx, cy=cy)

        rt_bbox = rotate_translate_bbox(bbox, theta=angle, cx=cx, cy=cy)
        rt_bbox_posecnn = posecnn_bbox
        if posecnn_bbox is not None:
            rt_bbox_posecnn = rotate_translate_bbox(posecnn_bbox,
                                                    theta=angle,
                                                    cx=cx,
                                                    cy=cy)
        rt_poses = rotate_translate_pose(poses,
                                         theta=angle,
                                         fx=fx,
                                         fy=fy,
                                         cx=cx,
                                         cy=cy)
        rt_poses_posecnn = rotate_translate_pose(posecnn_poses,
                                                 theta=angle,
                                                 fx=fx,
                                                 fy=fy,
                                                 cx=cx,
                                                 cy=cy)
        if not isinstance(rt_bbox, list):
            rt_bbox = list(rt_bbox)
        if posecnn_bbox is not None:
            if not isinstance(rt_bbox_posecnn, list):
                rt_bbox_posecnn = list(rt_bbox_posecnn)

        if not isinstance(rt_poses, list):
            rt_poses = list(rt_poses)
        if not isinstance(rt_poses_posecnn, list):
            rt_poses_posecnn = list(rt_poses_posecnn)

        if self.add_jitter:
            rt_img = self.jitter_transform(rt_img)
        if self.transform:
            rt_img = self.transform(rt_img)

        return (
            torch.Tensor(rt_img),
            torch.Tensor(np.array(rt_depth)),
            torch.Tensor(np.array(rt_label)),
            torch.Tensor(rt_bbox),
            torch.Tensor(rt_poses),
            torch.Tensor(rt_bbox_posecnn),
            torch.Tensor(rt_poses_posecnn),
        )



def voxel_grid_filter(points, leafSize):
    p = pcl.PointCloud()
    p.from_array(points)
    sor = p.make_voxel_grid_filter()
    sor.set_leaf_size(leafSize, leafSize, leafSize)
    cloud_filtered = sor.filter()
    return cloud_filtered


def outlier_filter(points, meanK, std):
    p = pcl.PointCloud(points)
    # p.from_array(points)
    fil = p.make_statistical_outlier_filter()
    fil.set_mean_k(meanK)
    fil.set_std_dev_mul_thresh(std)
    filtered_target = fil.filter()
    return filtered_target


def get_keyframe_list(data_root_path):
    lis = []
    file_path = os.path.join(data_root_path, "keyframe.txt")
    with open(file_path, "r") as file:
        lines = file.readlines()
    for i in range(0, len(lines)):
        lis.append(lines[i].strip())
    return lis


def get_classes(data_root_path):
    classes_file = os.path.join(data_root_path, "classes.txt")
    classes = {}
    with open(classes_file, "r") as file:
        lines = file.readlines()

    for i in range(0, len(lines)):
        classes[str(lines[i]).strip()] = i + 1

    return classes


def load_object_points(data_root_path, dense=False, new_num=100000):
    classes = get_classes(data_root_path)
    points = [[] for _ in range(0, len(classes))]
    num = np.inf

    for i in range(0, len(classes)):
        if dense:
            point_file = os.path.join(
                data_root_path,
                "models",
                list(classes.keys())[list(classes.values()).index(i + 1)],
                "points1.xyz",
            )
        else:
            point_file = os.path.join(
                data_root_path,
                "models",
                list(classes.keys())[list(classes.values()).index(i + 1)],
                "points.xyz",
            )
        assert os.path.exists(point_file), "Path does not exist: {}".format(
            point_file)
        points[i] = np.loadtxt(point_file)
        if points[i].shape[0] < num:
            num = points[i].shape[0]

    if dense:  # Fix this later.
        points_all = np.zeros((len(classes), new_num, 3), dtype=np.float32)
        for i in range(0, len(classes)):
            indices = np.random.randint(0, points[i].shape[0], new_num)
            points_all[i, :, :] = points[i][indices, :]
    else:
        points_all = np.zeros((len(classes), num, 3), dtype=np.float32)
        for i in range(0, len(classes)):
            points_all[i, :, :] = points[i][:num, :]

    return points, points_all


def rotate_translate_image(image, theta=0, cx=0, cy=0, fillcolor=100):
    """
    Rotates and translates PIL Image
    :param image: Input image in PIL Image format
    :param theta: Angle to be rotated by in degrees
    :param cx: Translation in x axis in pixels
    :param cy: Translation in y axis in pixels
    :return: Rotated and translated image
    """
    if type(image).__module__ == np.__name__:
        image = Image.fromarray(image)

    rot_trans_image = image.rotate(angle=theta,
                                   translate=(cx, cy),
                                   fillcolor=fillcolor)
    return rot_trans_image


def rotate_translate_pose(poses, theta=0, fx=0, fy=0, cx=0, cy=0):
    """
    Rotates and Translates poses
    :param poses: Ground truth poses for all the objects in the image
    :param theta: Angle to be rotated by in degrees
    :param fx: Focal length x
    :param fy: Focal length y
    :param cx: Translation in x axis in pixels
    :param cy: Translation in y axis in pixels
    :return: Rotated and Translated poses
    """
    if isinstance(poses, list):
        poses = np.array(poses, dtype=float)

    temp_pose = poses
    if poses.shape[2] > 4:
        poses = []
        for i in range(len(temp_pose)):
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = quat2mat(temp_pose[i][0][:4])
            RT[:, 3] = temp_pose[i][0][4:7]
            poses.append(RT)
        poses = np.array(poses, dtype=float)
    " poses are already in num x 3 x 4 format"
    rot_trans_poses = np.matmul(rot_z(-theta), poses)
    "Copies Tz to and then multiplies by c/f for x and y"
    "Tx' = Tx + cx*Tz/fx and Ty' = Ty + cy*Tz/fy"

    c_f = np.array([cx / fx, cy / fy], dtype=float)
    poses_hom = np.tile(np.eye(4, 4), (poses.shape[0], 1, 1))
    trans_mat = np.tile(np.eye(3, 4), (poses.shape[0], 1, 1))
    poses_hom[:, 0:3, :] = rot_trans_poses
    trans_mat[:, 0:2, 3] = np.tile(poses_hom[:, 2, 3],
                                   (2, 1)).transpose(1, 0) * c_f
    rot_trans_poses = np.matmul(trans_mat, poses_hom)
    return rot_trans_poses


def rotate_translate_bbox(bbox, theta=0, cx=0, cy=0, h=480, w=640):
    """
    Rotates and Translates ground truth amodal bounding box to give new
    *amodal* bounding box.
    :param bbox: Input bbox (numpy array) with x1,y1,x2,y2
    :param theta: Angle to be rotated in degrees
    :param cx: Translation in x axis in pixels
    :param cy: Translation in y axis in pixels
    :return: Roated and Translated bbox (array)
    """

    if isinstance(bbox, list):
        bbox = np.array(bbox)

    theta = np.pi * (-theta) / 180  # anti clockwise
    c = np.cos(theta)
    s = np.sin(theta)
    matrix = np.array([[c, -s, cx], [s, c, cy]], dtype=float)

    num = bbox.shape[0]
    bbox_center = np.tile([320, 240], (num * 4, 1))

    bbox = np.concatenate(
        [
            bbox,
            bbox[:, 0].reshape(num, 1),
            bbox[:, 3].reshape(num, 1),
            bbox[:, 2].reshape(num, 1),
            bbox[:, 1].reshape(num, 1),
        ],
        axis=1,
    )
    bbox_hom = np.ones((num * 4, 3), dtype=float)
    bbox_hom[:, 0:2] = bbox.reshape(num * 4, 2) - bbox_center
    rot_trans_bbox_temp = (
        np.matmul(matrix, bbox_hom.transpose()).transpose() +
        bbox_center).reshape(num, 8)

    min_x = np.clip(np.min(rot_trans_bbox_temp[:, 0::2], axis=1), 0, w)
    max_x = np.clip(np.max(rot_trans_bbox_temp[:, 0::2], axis=1), 0, w)
    min_y = np.clip(np.min(rot_trans_bbox_temp[:, 1::2], axis=1), 0, h)
    max_y = np.clip(np.max(rot_trans_bbox_temp[:, 1::2], axis=1), 0, h)

    rot_trans_bbox = np.array([min_x, min_y, max_x, max_y]).transpose(1, 0)
    return rot_trans_bbox


def rot_x(phi):
    """
    Rotation about the x-axis.
    :param phi: In degrees
    :return: matrix
    """
    phi = np.pi * phi / 180
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(theta):
    """
    Rotation about the x-axis.
    :param theta: In degrees
    :return: matrix
    """
    theta = np.pi * theta / 180
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def rot_z(psi):
    """
    Rotation about the x-axis.
    :param psi: In degrees
    :return: matrix
    """
    psi = np.pi * psi / 180
    c = np.cos(psi)
    s = np.sin(psi)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
