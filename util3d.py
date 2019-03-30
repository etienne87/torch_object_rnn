#!/usr/bin/python

# This script is for pedagogic purpose, about cameras projections from world & back
#
# Copyright: (c) 2017-2018 Chronocam

import numpy as np
import cv2



class Cube(object):
    def __init__(self, dims, dtype=np.float32):
        self.points = np.array([[],
                               [],
                               [],
                               [],
                               [],
                               [],
                               [],
                               []
                               ], dtype=dtype)



# define plane parallel to the ground
def make_xy_plane(npts_x=5, npts_z=10, x_min=-10, x_max=10, z_min=0, z_max=50, height=0):
    n1 = npts_z
    n2 = npts_x
    z_step = float(z_max - z_min) / n1
    x_step = float(x_max - x_min) / n2
    n = n1 * n2
    objp = np.zeros((n, 3), np.float32)
    mg = np.mgrid[0:z_max:z_step, x_min:x_max:x_step].T.reshape(-1, 2)
    objp[:, 0] = mg[:, 1]
    objp[:, 2] = mg[:, 0]
    objp[:, 1] = height
    return objp

def make_cube(n=50, x_min=-1, x_max=1, y_min=-1, y_max=1, z_min=0, z_max=5):
    z_step = float(z_max - z_min) / n
    y_step = float(y_max - y_min) / n
    x_step = float(x_max - x_min) / n
    objp = np.mgrid[x_min:x_max:x_step, y_min:y_max:y_step, z_min:z_max:z_step].T.reshape(-1, 3)
    return objp

def dist2undist_cv(pts, K, dist_coeffs):
    im_grid_cv = np.expand_dims(pts, axis=0)
    rect_grid_cv = cv2.undistortPoints(im_grid_cv, K, dist_coeffs)[0]
    ones = np.ones((rect_grid_cv.shape[0], 1), dtype=rect_grid_cv.dtype)
    rect_grid_cv = np.concatenate((rect_grid_cv, ones), axis=1)
    return rect_grid_cv


# here you do want to use the rotation matrix because its transpose is the inverse
def cam2world(camRays, R):
    return camRays.dot(R)


def filter_2d(pts, height, width):
    ids = np.where((pts[:, 0] >= 0) & (pts[:, 0] < width) & (pts[:, 1] >= 0) & (pts[:, 1] < height))
    return ids

def filter_3d(pts, tvec):
    ids = np.where((pts[:, 1] <= 0) &
                   (pts[:, 2] >= tvec[2])) #points behind the camera
    return pts[ids]

def world_to_img(objp, rvec, tvec, K, dist_coeffs, height, width):
    if rvec.size == 3:
        R = cv2.Rodrigues(rvec)[0]
    else:
        R = rvec
    T = -R.dot(tvec)
    #filter points that are behind the camera
    objp = filter_3d(objp, tvec)
    if len(objp) == 0:
        return np.array([])

    img_grid = cv2.projectPoints(objp, R, T, K, dist_coeffs)[0].squeeze().reshape(-1, 2)
    ids = filter_2d(img_grid, height, width)
    objp = objp[ids]
    img_grid = img_grid[ids]
    return img_grid.astype(np.int32)


def draw_points(grid, img, color=(255,0,0)):
    for i in range(grid.shape[0]):
        pt = (int(grid[i, 0]), int(grid[i, 1]))
        cv2.circle(img, pt, 0, color, 3)


def show_plane_animation(objp, rvec, tvec, K, dist, axis=0):
    img = np.zeros((240, 304, 3), dtype=np.uint8)
    for angle in np.arange(-np.pi / 16, np.pi / 16, np.pi / 2000):
        img[...] = 128
        rvec[axis] = angle
        grid, _ = world_to_img(objp, rvec, tvec, K, dist)

        for i in range(grid.shape[0]):
            pt = (int(grid[i, 0]), int(grid[i, 1]))
            cv2.circle(img, pt, 0, (0, 0, 255), 3)
        cv2.imshow('img', img)
        cv2.waitKey(10)


def get_square(cube2d):
    if len(cube2d) == 0:
        return None, None
    xmin, ymin = cube2d.min(axis=0).tolist()
    xmax, ymax = cube2d.max(axis=0).tolist()
    return (xmin, ymin), (xmax, ymax)

OK = 0
TOOSMALL = 1
TOOBIG = 2
TOP = 3
BOTTOM = 4
LEFT = 5
RIGHT = 6


if __name__ == '__main__':

    dtype = np.float32
    height, width = 480, 640
    K = np.array([[300, 0, width/2],
                  [0, 300, height/2],
                  [0, 0, 1]], dtype=dtype)

    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype).reshape(5, )

    plane = make_xy_plane(npts_x=3, npts_z=100, x_min=-10, x_max=10, z_min=0, z_max=7)
    cube = make_cube(n=10, x_min=-1, x_max=1, y_min=-2, y_max=0.0, z_min=1.2, z_max=1.2+2)

    rvec = np.array([0, 0, 0], dtype=dtype)  # PITCH, YAW, ROLL
    tvec = np.array([0, -1, -2.0], dtype=dtype)

    last_img = np.full((height, width, 3), 127, dtype=np.uint8)
    img = np.full((height, width, 3), 127, dtype=np.uint8)

    cv2.namedWindow("image")

    while 1:
        last_img[...] = img
        img[...] = 127
        rvec = rvec*0.999 + np.random.randn(3) * 0.001
        tvec = tvec*0.999 + np.random.randn(3) * 0.001

        rvec[0] += 0.05
        rvec[1] += 0.1
        rvec[2] += 0.1

        cube2d = world_to_img(cube, rvec, tvec, K, dist, height, width) #TODO: put on GPU
        plane2d = world_to_img(plane, rvec, tvec, K, dist, height, width) #TODO: put on GPU


        draw_points(plane2d, img, (255, 0, 255))
        draw_points(cube2d, img, (0, 0, 255))


        diff = img.astype(np.float32)/255.0-last_img.astype(np.float32)/255.0 + 0.5

        tl, br = get_square(cube2d)
        if tl:
            cv2.rectangle(diff, tl, br, (255,0,0), 2)

        cv2.imshow('image', img)
        cv2.imshow('diff', diff)
        key = cv2.waitKey(5)
        if key == ord('q'):
            #rvec[1] -= np.pi/16
            tvec[0] -= 0.1
        elif key == ord('d'):
            #rvec[1] += np.pi/16
            tvec[0] += 0.1
        elif key == ord('s'):
            # rvec[0] -= np.pi/16
            tvec[2] -= 0.1
        elif key == ord('z'):
            #rvec[0] += np.pi / 16
            tvec[2] += 0.5
        elif key == ord('w'):
            # rvec[0] -= np.pi/16
            tvec[1] -= 0.1
        elif key == ord('x'):
            #rvec[0] += np.pi / 16
            tvec[1] += 0.1
        if key == 27:
            break
