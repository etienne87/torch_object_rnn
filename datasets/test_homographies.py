from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
from pycocotools.coco import COCO



def moving_average(item, alpha):
    return (1-alpha)*item + alpha * np.random.randn(*item.shape)


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


def show_voyage(img, anns):
    height, width = img.shape[:2]

    voyage = PlanarVoyage(height, width)

    prev_img = img.astype(np.float32)


    mask = coco.annToMask(anns[0])
    for i in range(len(anns)):
        mask += coco.annToMask(anns[i])
    mask_rgb = cv2.applyColorMap((mask * 30) % 255, cv2.COLORMAP_HSV) * (mask > 0)[:, :, None].repeat(3, 2)
    mask_rgb = mask_rgb.astype(np.float32)/255.0


    d = 1
    t = 0

    while 1:

        G_0to2 = voyage()

        out = cv2.warpPerspective(img, G_0to2, (width, height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32)
        out = (out - out.min()) / (out.max() - out.min())


        out_mask = cv2.warpPerspective(mask_rgb, G_0to2, (width, height),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP).astype(np.float32)

        diff = out - prev_img

        # Remove some events


        # flow = get_flow(G_0to2, height, width)
        # flow = (flow-flow.min())/(flow.max()-flow.min())
        # viz_flow = flow_viz(flow)
        # cv2.imshow('flow', viz_flow)
        # im = (prev_img*255).astype(np.uint8)
        # ts = compute_timesurface(im, flow, diff).clip(0)
        # ts = filter_outliers(ts)
        # ts = (ts-ts.min())/(ts.max()-ts.min())
        # cv2.imshow('ts', ts)





        # Salt-and-Pepper
        # diff += (np.random.rand(height, width)[:,:,None].repeat(3,2) < 0.00001)/2
        # diff -= (np.random.rand(height, width)[:,:,None].repeat(3,2) < 0.00001) / 2

        diff *= np.random.rand(height, width, 3) < 0.9
        diff = viz_diff(diff) + out_mask / 3


        cv2.imshow("diff", diff)
        cv2.imshow("out", out)
        key = cv2.waitKey(0)
        if key == 27:
            break
        prev_img = out

        t += 1


if __name__ == '__main__':
    import glob
    #imgs = glob.glob("/home/etienneperot/workspace/data/coco/images/train2017/"+"*.jpg")

    dataDir = '/home/etienneperot/workspace/data/coco'
    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)

    catIds = coco.getCatIds(catNms=['person', 'car'])
    imgIds = coco.getImgIds(catIds=catIds)

    while 1:
        img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
        file_name = os.path.join(dataDir, 'images', dataType, img['file_name'])
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        image = cv2.imread(file_name)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])


        show_voyage(image, anns)
