#!/usr/bin/python
# This script runs a random 3d animation & can retrieve:
# - detection
# - flow
# - zbuffer
#
# Copyright: (c) 2017-2018 Chronocam

import numpy as np
import cv2
import time

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

def plane_intersect(rays,tvec,p0=np.array([0,0,0]),normal=np.array([0,1,0])):
    n = normal.T
    l = rays
    l0 = tvec
    a = (p0 - l0).dot(n)
    b = l.dot(n) + 1e-20
    d = a / b
    d = np.expand_dims(d,axis=1)
    p = d*l + l0
    return p

def make_cube(n=50, x_min=-1, x_max=1, y_min=-1, y_max=1, z_min=0, z_max=5):
    # z_step = float(z_max - z_min) / n
    # y_step = float(y_max - y_min) / n
    # x_step = float(x_max - x_min) / n
    # objp = np.mgrid[x_min:x_max:x_step, y_min:y_max:y_step, z_min:z_max:z_step].T.reshape(-1, 3)

    objp = np.array([
                    [x_min, y_min, z_min],
                    [x_min, y_min, z_max],
                    [x_min, y_max, z_max],
                    [x_min, y_max, z_min],
                    [x_max, y_min, z_min],
                    [x_max, y_min, z_max],
                    [x_max, y_max, z_max],
                    [x_max, y_max, z_min]], dtype=np.float32)

    return objp

def add_points_on_faces(cube):
    planes = [[0, 1, 2, 3],
              [4, 5, 6, 7],
              [0, 4, 7, 3],
              [1, 5, 6, 2],
              [0, 1, 5, 4],
              [3, 2, 6, 7]]
    points = []
    for plane in enumerate(planes):
        points = np.meshgrid()


def draw_cube_faces(img, cube, zbuffer):
    planes = [[0, 1, 2, 3],
              [4, 5, 6, 7],
              [0, 4, 7, 3],
              [1, 5, 6, 2],
              [0, 1, 5, 4],
              [3, 2, 6, 7],
              ]
    cmap = cv2.applyColorMap(np.arange(0, 255, 255/8, dtype=np.uint8), cv2.COLORMAP_AUTUMN).tolist()
    planes2 = []
    for i, plane in enumerate(planes):
        zavg = zbuffer[plane].mean(0)
        clr = cmap[i][0] + [1.0-zavg/3.0]
        planes2.append((zavg, cube[plane], clr))
    planes2 = sorted(planes2, key=lambda x:x[0], reverse=False)
    for zm, plane, clr in planes2:
        cv2.fillConvexPoly(img, plane, clr)


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
    ids = np.where((pts[:, 1] <= 2) &
                   (pts[:, 2] >= tvec[2])) #points behind the camera
    return pts[ids]

def world_to_img(objp, rvec, tvec, K, dist_coeffs, height, width):
    if rvec.size == 3:
        R = cv2.Rodrigues(rvec)[0]
    else:
        R = rvec
    T = -R.dot(tvec)
    img_grid = cv2.projectPoints(objp, R, T, K, dist_coeffs)[0].squeeze().reshape(-1, 2).astype(np.int32)
    ids = filter_2d(img_grid, height, width)
    img_grid2 = img_grid[ids]
    zbuffer = 1./(objp[:, 2]+0.1)
    return img_grid2, img_grid, zbuffer

def rotate(objp, cog, rspeed):
    R = cv2.Rodrigues(rspeed)[0]
    xyz = (objp-cog).dot(R.T) + cog
    return xyz


def draw_points(grid, img, zbuffer, color=(255,0,0)):
    for i in range(grid.shape[0]):
        pt = (int(grid[i, 0]), int(grid[i, 1]))
        cv2.circle(img, pt, 1, color, 3)


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


def draw_flow(img, old, new):
    cmap = cv2.applyColorMap(np.arange(255, dtype=np.uint8), cv2.COLORMAP_HSV).tolist()

    for i in range(new.shape[0]):
        pt1 = (old[i, 0], old[i, 1])
        pt2 = (new[i, 0]+1, new[i, 1]+1)
        angle = np.arctan2(pt2[1]-pt1[1], pt2[0]-pt1[0])
        angle = abs(int(min(angle, np.pi-0.01)*255/np.pi))
        color = cmap[angle][0]
        cv2.arrowedLine(img, pt1, pt2, color, 2)


OK = 0
TOOSMALL = 1
TOOBIG = 2
TOP = 3
BOTTOM = 4
LEFT = 5
RIGHT = 6


if __name__ == '__main__':

    dtype = np.float64
    height, width = 480, 640
    K = np.array([[300, 0, width/2],
                  [0, 300, height/2],
                  [0, 0, 1]], dtype=dtype)

    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=dtype).reshape(5, )

    plane = make_xy_plane(npts_x=3, npts_z=100, x_min=-10, x_max=10, z_min=0, z_max=7)
    #cube = make_cube(n=5, x_min=-1, x_max=1, y_min=-2, y_max=0, z_min=1.2, z_max=1.2+2)
    cube = make_cube(n=5, x_min=-1, x_max=1, y_min=-2, y_max=0, z_min=1.2, z_max=1.2 + 2)
    cube_cog = np.array([0,-1,2.2], dtype=dtype)

    rvec = np.array([0, 0, 0], dtype=dtype)  # PITCH, YAW, ROLL
    tvec = np.array([-1, -4, -4], dtype=dtype)

    last_img = np.full((height, width, 3), 127, dtype=np.uint8)
    img = np.full((height, width, 3), 127, dtype=np.uint8)

    old_cube2d = None
    old_plane2d = None

    cv2.namedWindow("image")

    rspeed = np.random.randn(3) * 0.01
    tspeed = np.random.randn(3) * 0.1
    stop_iter = 0

    while 1:



        if stop_iter == 0:
            last_img[...] = img
            img[...] = 127

        rspeed = rspeed*0.99 + np.random.randn(3)*0.01
        tspeed = tspeed*0.99 + np.random.randn(3)*0.01

        if stop_iter == 0:
            cube = rotate(cube, cube_cog, rspeed)
            tvec += tspeed
        else:
            stop_iter -= 1


        cube2d, fullcube2d, cube2dz = world_to_img(cube, rvec, tvec, K, dist, height, width) #TODO: put on GPU

        plane2d, fullplane2d, planez = world_to_img(plane, rvec, tvec, K, dist, height, width) #TODO: put on GPU
        start = time.time()

        draw_cube_faces(img, fullcube2d, cube2dz)
        #draw_points(plane2d, img, zbuffer=planez, color=(255, 0, 255))
        draw_points(cube2d, img, zbuffer=cube2dz, color=(0, 0, 255))

        diff = img.astype(np.float32)/255.0-last_img.astype(np.float32)/255.0 + 0.5
        tl, br = get_square(fullcube2d)

        show = img.copy()

        if old_cube2d is not None:
            draw_flow(show, old_cube2d, fullcube2d)

        print(time.time() - start, ' s')


        old_cube2d = fullcube2d
        old_plane2d = fullplane2d

        if tl:
            color = (0,0,0)
            cv2.rectangle(show, tl, br, color, 2)
            cv2.putText(show, "a cube!", br, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            tl2, br2 = get_square(fullcube2d)
            fullwidth = br2[0]-tl2[0]
            fullheight = br2[1]-tl2[1]

            min_width = 64
            min_height = 64
            x1, y1 = tl
            x2, y2 = br
            flags = {}


            if x1 <= 0:
                flags[LEFT] = 1
                tspeed[0] = -abs(tspeed[0])

            if x2 >= width - 1:
                flags[RIGHT] = 1
                tspeed[0] = abs(tspeed[0])

            if y1 <= 0:
                flags[TOP] = 1
                tspeed[1] = -abs(tspeed[1])

            if y2 >= height -1:
                flags[BOTTOM] = 1
                tspeed[1] = abs(tspeed[1])

            if fullwidth <= min_width or fullheight <= min_height:
                flags[TOOSMALL] = 1
                tspeed[2] = abs(tspeed[2])

            if fullwidth >= width - 2 or fullheight >= height - 2:
                flags[TOOBIG] = 1
                tspeed[2] = -abs(tspeed[2])

        if np.random.rand() < 0.01:
            stop_iter = 3



        cv2.imshow('image', show)
        key = cv2.waitKey(0)
        if key == 27:
            break
