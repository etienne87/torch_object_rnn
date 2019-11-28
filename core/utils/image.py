from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import random


def flip(img):
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3,))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def computeC2MC1(R_0to1, tvec_0to1, R_0to2, tvec_0to2):
    R_1to2 = R_0to2.dot(R_0to1.T)
    tvec_1to2 = R_0to2.dot(-R_0to1.T.dot(tvec_0to1)) + tvec_0to2
    return R_1to2, tvec_1to2

def generate_homography(rvec1, tvec1, rvec2, tvec2, nt, K, Kinv, d):
    R_0to1 = cv2.Rodrigues(rvec1)[0].transpose()
    tvec_0to1 = np.dot(-R_0to1, tvec1.reshape(3, 1))

    R_0to2 = cv2.Rodrigues(rvec2)[0].transpose()
    tvec_0to2 = np.dot(-R_0to2, tvec2.reshape(3, 1))

    #view 0to2
    nt1 = R_0to1.dot(nt.T).reshape(1, 3)
    H_0to2 = R_0to2 - np.dot(tvec_0to2.reshape(3, 1), nt1) / d
    G_0to2 = np.dot(K, np.dot(H_0to2, Kinv))


    #view 1to2
    # R_1to2, tvec_1to2 = computeC2MC1(R_0to1, tvec_0to1, R_0to2, tvec_0to2)
    # H_1to2 = R_1to2 - np.dot(tvec_1to2.reshape(3, 1), nt1) / d
    # G_1to2 = np.dot(K, np.dot(H_1to2, Kinv))
    return G_0to2


def viz_diff(diff):
    diff = diff.clip(diff.mean() - 3 * diff.std(), diff.mean() + 3 * diff.std())
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    return diff

def gradient(plane, k=3):
    gx, gy = cv2.spatialGradient(plane, k, k)
    return np.concatenate([gx[...,None], gy[...,None]], axis=2).astype(np.float32)/255.0

def compute_timesurface(img, flow, diff):
    if img.ndim == 3:
        gxys = [gradient(img[...,i]) for i in range(3)]
        gxy = np.maximum(np.maximum(gxys[0], gxys[1]), gxys[2])
    else:
        gxy = gradient(img)

    if diff.ndim == 3:
        diff = diff.mean(axis=2)

    # normalize
    gxy = (gxy - gxy.min()) / (gxy.max() - gxy.min())

    gflow = (gxy * flow).sum(axis=2)
    time = diff / (1e-7 + gflow)
    return time


def get_flow(homography, height, width):
    x, y = np.meshgrid(np.linspace(-width/2, width/2, width), np.linspace(-height/2, height/2, height))
    x, y = x[:, :, None], y[:, :, None]
    o = np.ones_like(x)
    xy1 = np.concatenate([x, y, o], axis=2)
    xyn = xy1.reshape(height * width, 3)
    xyn2 = xyn.dot(homography)
    denom = (1e-7 + xyn2[:,2][:,None])
    xy2 = xyn2[:,:2] / denom
    xy2 = xy2.reshape(height, width, 2)
    flow = xy2-xy1[...,:2]
    flow[:,:,0] /= width
    flow[:,:,1] /= height
    return flow


def filter_outliers(input_val, num_std=2):
    val_range = num_std * input_val.std()
    img_min = input_val.mean() - val_range
    img_max = input_val.mean() + val_range
    normed = np.clip(input_val, img_min, img_max)  # clamp
    return normed

def flow_viz(flow):
    h, w, c = flow.shape
    hsvImg = np.zeros((h, w, 3), dtype=np.uint8)
    hsvImg[..., 1] = 255
    _, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
    hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
    hsvImg[..., 1] = 255
    hsvImg[..., 2] = 255
    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)
    return rgbImg


def moving_average(item, alpha):
    return (1-alpha)*item + alpha * np.random.randn(*item.shape)


class PlanarVoyage(object):
    def __init__(self, height, width):
        self.K = np.array([[width / 2, 0, width / 2],
                      [0, width / 2, height / 2],
                      [0, 0, 1]], dtype=np.float32)
        self.Kinv = np.linalg.inv(self.K)

        self.rvec1 = np.array([0, 0, 0], dtype=np.float32)
        self.tvec1 = np.array([0, 0, 0], dtype=np.float32)
        self.nt = np.array([0, 0, -1], dtype=np.float32).reshape(1, 3)

        self.rvec_amp = np.random.rand(3) * 0.25
        self.tvec_amp  = np.random.rand(3) * 0.5

        self.rvec_speed = np.random.choice([1e-1,1e-2,1e-3])
        self.tvec_speed = np.random.choice([1e-1, 1e-2, 1e-3])

        self.rvec_amp[2] = 0.0

        self.tshift = np.random.randn(3)
        self.rshift = np.random.randn(3)
        self.d = 1
        self.time = 0

    def __call__(self):
        self.tshift = moving_average(self.tshift, 1e-4)
        self.rshift = moving_average(self.rshift, 1e-4)
        rvec2 = self.rvec_amp * np.sin(self.time * self.rvec_speed + self.rshift)
        tvec2 = self.tvec_amp * np.sin(self.time * self.tvec_speed + self.tshift)
        G_0to2 = generate_homography(self.rvec1, self.tvec1, rvec2, tvec2, self.nt, self.K, self.Kinv, self.d)
        G_0to2 /= G_0to2[2,2]
        self.time += 1
        return G_0to2


def wrap_boxes(boxes, height, width):
    """repeats boxes in 8 cells around the image
    """
    allbox = [boxes]
    for y in [-height, 0, height]:
        for x in [-width, 0, width]:
            if x == 0 and y == 0:
                continue
            shift = boxes.copy()
            shift[:, [0, 2]] += x
            shift[:, [1, 3]] += y
            allbox.append(shift)
    return np.concatenate(allbox, 0)


def clamp_boxes(boxes, height, width):
    boxes[:, [0, 2]] = np.maximum(np.minimum(boxes[:, [0, 2]], width), 0)
    boxes[:, [1, 3]] = np.maximum(np.minimum(boxes[:, [1, 3]], height), 0)
    return boxes


def discard_too_small(boxes, min_size=30):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    ids = np.where((w > min_size)*(h > min_size))
    return boxes[ids]


def point_list_toboxes(pts):
    """
    boxes have been rotated sheared, etc, we need horizontal boxes in xy w h format
    args :
        - pts (ndarray n x 1 x 2) points in pixel coordinates in the sisley space
    return :
        - (ndarray n/4 x 4) boxes in x,y,w,h format
    """
    if pts.shape[0] > 0 and pts.shape[0] % 4 != 0:
        raise ValueError('input must be a list of corners points ! (divisible by 4 !)')
    boxes = np.empty((pts.shape[0] // 4, 4), dtype=pts.dtype)
    points = np.concatenate((pts[:pts.shape[0] // 4], pts[pts.shape[0] // 4:pts.shape[0] // 2],
                             pts[pts.shape[0] // 2:-pts.shape[0] // 4], pts[-pts.shape[0] // 4:]), axis=1)

    boxes[:, 0] = points[..., 0].min(axis=1)
    boxes[:, 1] = points[..., 1].min(axis=1)
    boxes[:, 2] = points[..., 0].max(axis=1)
    boxes[:, 3] = points[..., 1].max(axis=1)
    return boxes


def np_perspective_transform(pts, transform):
    """
    warp boxes with homography (similar to cv2.perspectiveTransform)
    :param boxes:
    :param transform:
    :return:
    """
    tmp = pts.dot(transform[:2, :]) + transform[2]
    tmp = tmp / (tmp[:, 2:3] + 1e-8)
    return tmp[:, :2]


def cv2_apply_transform_boxes(boxes, transform):
    """
       Applies
       :param boxes: list of N, 4 parameters
       :param transforms:
       :return:
    """
    if boxes.shape[0] == 0:
        return boxes

    pts = np.concatenate((boxes[:, :2], boxes[:, 2:], boxes[:, [0, 3]], boxes[:, [2, 1]]))

    pts = pts.reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, transform)
    boxes = point_list_toboxes(pts)
    return boxes

def cv2_clamp_filter_boxes(boxes, transform, imsize, extra_size_inv_ratio=None):
    """
    Clamping boxes
    :param boxes:
    :param transform:
    :param imsize: height and width
    :extra_size_inv_ratio: if not None the coordinates limits are extended by
     width // extra_size_inv_ratio and height // extra_size_inv_ratio for x and y respectively on the left, right, up, down
    :return:
    """
    h, w = imsize
    if extra_size_inv_ratio is None:
        extra_border_x = 0
        extra_border_y = 0
    else:
        extra_border_x = w // extra_size_inv_ratio
        extra_border_y = h // extra_size_inv_ratio
    frame = np.array([[-extra_border_x, -extra_border_y, w + extra_border_x, h + extra_border_y]], dtype=np.float32)
    frame = cv2_apply_transform_boxes(frame, transform)
    # clamp
    xmin, ymin, xmax, ymax = frame[0, 0], frame[0, 1], frame[0, 2], frame[0, 3]
    xmin, ymin, xmax, ymax = max(xmin, -extra_border_x), max(ymin, -extra_border_y),\
        min(xmax, imsize[1] - 1 + extra_border_x), min(ymax, imsize[0] - 1 + extra_border_y)

    out = boxes.copy()
    out[:, [0, 2]] = np.maximum(np.minimum(out[:, [0, 2]], xmax), xmin)
    out[:, [1, 3]] = np.maximum(np.minimum(out[:, [1, 3]], ymax), ymin)

    return out


def regiongrowing(gt, shift=1, discard_size=20):
    bbs=[]
    labels=[]
    h,w = gt.shape
    mask = np.zeros((h+2, w+2), np.uint8)
    idx=1
    for y in range(0,h,shift):
        for x in range(0,w,shift):
            if mask[y,x] == 0:
                flag = (8 | idx << 8)  | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
                bb = cv2.floodFill(gt,mask,(x,y),idx,0,0,flag)
                x1,y1,w1,h1 = bb[-1]
                x2 = x1+w1
                y2 = y1+h1
                #discard too small
                if w1 < discard_size or h1 < discard_size:
                    continue

                bbox = np.array([[x1,y1,x2,y2]])
                bbs.append(bbox)
                labels.append(gt[y,x])
                idx=idx+1
    return np.concatenate(bbs, 0), np.array(labels)


if __name__ == '__main__':
    heatmap = np.zeros((256, 256), dtype=np.float32)
    center = (128, 128)
    for sigma in range(2, 256):
        # img = draw_umich_gaussian(heatmap, center, sigma)
        img = draw_msra_gaussian(heatmap, center, sigma)

        cv2.imshow('img', img)
        cv2.waitKey(0)
