import torch
import torch.nn as nn

# from flownet2 import fl2_models


def weight_initialise(model):
    # Initialise with Kaiming normal
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight,
                                   mode='fan_out',
                                   nonlinearity='relu')
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal(m.weight,
                                   mode='fan_out',
                                   nonlinearity='relu')


def get_updated_pos(bbox, est, K):
    """
    This function takes in 2d bounding box, estimated pos and returns 3D
    :param bbox: the box coordinates in x1,y1,x2,y2
    :param est: estimated position del_x del_y z
    :param K: camera intrinsic parametes
    :return: Updated 3d pos
    """
    _b, _t, _, _ = K.shape
    K = K.reshape(-1, 3, 3).unsqueeze(1)
    num_repeat = est.shape[0] // (K.shape[0])
    _K = K.repeat(1, num_repeat, 1, 1)
    _K = _K.reshape(_b, _t, -1, 3, 3).view(-1, 3, 3)

    fx, fy, px, py = _K[:, 0, 0], _K[:, 1, 1], _K[:, 0, 2], _K[:, 1, 2]
    cx = (bbox[:, 1] + bbox[:, 3]) / 2
    cy = (bbox[:, 2] + bbox[:, 4]) / 2
    scale = est[:, 2].clone().detach().requires_grad_(requires_grad=False)
    output_t = torch.empty(est.shape).type_as(est)
    output_t[:, 0] = torch.mul(torch.div((cx + est[:, 0] - px), fx), scale)
    output_t[:, 1] = torch.mul(torch.div((cy + est[:, 1] - py), fy), scale)
    output_t[:, 2] = est[:, 2]
    return output_t.requires_grad_()


def get_updated_pos_reverse(bbox, est, K):
    """
    This function takes in 2d bounding box, estimated pos and returns 3D
    :param bbox: the box coordinates in x1,y1,x2,y2
    :param est: estimated position del_x del_y z
    :param K: camera intrinsic parametes
    :return: Updated 3d pos
    """
    num_repeat = est.shape[0] // K.shape[0]
    _K = K.transpose(1, 0).repeat(num_repeat, 1,
                                  1).transpose(1,
                                               0).contiguous().view(-1, 3,
                                                                    3).float()

    fx, fy, px, py = _K[:, 0, 0], _K[:, 1, 1], _K[:, 0, 2], _K[:, 1, 2]
    cx = (bbox[:, 1] + bbox[:, 3]) / 2
    cy = (bbox[:, 2] + bbox[:, 4]) / 2
    scale = est[:, 2].clone().detach().requires_grad_(requires_grad=False)
    ind = (scale == 0).nonzero()
    output_t = torch.empty(est.shape).type_as(est)
    output_t[:, 0] = torch.div(torch.mul(est[:, 0], fx), scale) + px - cx
    output_t[:, 1] = torch.div(torch.mul(est[:, 1], fy), scale) + py - cy
    output_t[:, 2] = est[:, 2]
    output_t[ind] = 0
    return output_t


def iou(pred, target):
    """
    Calculates intersection over union for every class in every batch.
    Ignores background
    Returns NaN when no target is present.
    :param pred: B x n_classes prediction
    :param target: B x n_classes ground truth
    :return: B x n_classes-1 ious
    """
    target = make_one_hot(target.unsqueeze(1).long(), 22).double()[:, 1::, :]
    _, pred_ = pred.max(1)
    pred_ = make_one_hot(pred_.unsqueeze(1), 22).double()[:, 1::, :]
    b = pred_.size(0)
    c = pred_.size(1)

    intersection = (pred_ * target.double()).reshape(b, c, -1).sum(2)
    union = (pred_.reshape(b, c, -1)).sum(2) + \
            (target.reshape(b, c, -1)).sum(2) - intersection
    ious = intersection / union

    # ious will contain NaNs when no target is present.
    return ious


def make_one_hot(labels, C=2):
    """
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    From Jacob Kimmel's blog post

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    """
    one_hot = torch.cuda.FloatTensor(labels.size(0), C, labels.size(2),
                                     labels.size(3)).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def mat2quat(rot):
    "Converts rotation matrix to quaternions"
    q = torch.zeros((4))
    tr = rot[0, 0] + rot[1, 1] + rot[2, 2]

    if tr > 0:
        S = (tr + 1.0).sqrt() * 2
        q[0] = 0.25 * S
        q[1] = (rot[2, 1] - rot[1, 2]) / S
        q[2] = (rot[0, 2] - rot[2, 0]) / S
        q[3] = (rot[1, 0] - rot[0, 1]) / S
    elif (rot[0, 0] > rot[1, 1]) & (rot[0, 0] > rot[2, 2]):
        S = (1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]).sqrt() * 2
        q[0] = (rot[2, 1] - rot[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (rot[0, 1] + rot[1, 0]) / S
        q[3] = (rot[0, 2] + rot[2, 0]) / S
    elif rot[1, 1] > rot[2, 2]:
        S = (1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]).sqrt() * 2
        q[0] = (rot[0, 2] - rot[2, 0]) / S
        q[1] = (rot[0, 1] + rot[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (rot[1, 2] + rot[2, 1]) / S
    else:
        S = (1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]).sqrt() * 2
        q[0] = (rot[1, 0] - rot[0, 1]) / S
        q[1] = (rot[0, 2] + rot[2, 0]) / S
        q[2] = (rot[1, 2] + rot[2, 1]) / S
        q[3] = 0.25 * S

    return q.to(rot.device)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: First three coeff of quaternion of rotation. \
            Fourth is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    if quat.norm() < 0.01:
        print("quat2mat ", quat.numpy(), quat.norm().numpy())
    norm_quat = quat / quat.norm()
    w, x, y, z = norm_quat[0], norm_quat[1], norm_quat[2], norm_quat[3]

    x2, y2, z2 = x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        1 - 2 * y2 - 2 * z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        1 - 2 * x2 - 2 * z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        1 - 2 * x2 - 2 * y2
    ],
                         dim=0)
    rotMat = rotMat.reshape(3, 3).type_as(quat).requires_grad_()
    return rotMat


def transfrom_pts(R_est, t_est, R_gt, t_gt, pts):
    """
    reprojection error.
    :param K intrinsic matrix
    :param R_est, t_est: Estimated pose (3x3 rot matrix and 3x1 trans vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    if t_est is None:
        pts_est = transform_pts_R(pts, R_est)
        pts_gt = transform_pts_R(pts, R_gt)
    else:
        pts_est = transform_pts_Rt(pts, R_est, t_est)
        pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    return pts_gt, pts_est


def transform_pts_R(pts, R):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.size(1) == 3
    pts_t = torch.mm(R, pts.t()).type_as(pts).requires_grad_()
    return pts_t.t()


def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.size(1) == 3
    pts_t = torch.add(torch.mm(R, pts.t()),
                      torch.reshape(t, (3, 1))).type_as(pts).requires_grad_()
    return pts_t.t()


def reproj(K, R_est, t_est, R_gt, t_gt, pts):
    """
    reprojection error.
    :param K intrinsic matrix
    :param R_est, t_est: Estimated pose (3x3 rot matrix and 3x1 trans vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    device = R_est.device

    K = K.float().requires_grad_()
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    pixels_est = torch.mm(K, pts_est.t()).to(device).requires_grad_()
    pixels_est = pixels_est.t()
    pixels_gt = torch.mm(K, pts_gt.t()).to(device).requires_grad_()
    pixels_gt = pixels_gt.t()

    t1 = torch.clamp(torch.div(pixels_est[:, 0], pixels_est[:, 2]),
                     min=0,
                     max=639)
    t2 = torch.clamp(torch.div(pixels_est[:, 1], pixels_est[:, 2]),
                     min=0,
                     max=479)
    est = torch.cat((t1.unsqueeze(0), t2.unsqueeze(0)), dim=0)

    t1 = torch.clamp(torch.div(pixels_gt[:, 0], pixels_gt[:, 2]),
                     min=0,
                     max=639)
    t2 = torch.clamp(torch.div(pixels_gt[:, 1], pixels_gt[:, 2]),
                     min=0,
                     max=479)
    gt = torch.cat((t1.unsqueeze(0), t2.unsqueeze(0)), dim=0)

    return gt, est
