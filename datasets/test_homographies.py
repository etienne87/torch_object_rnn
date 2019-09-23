from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import math


def offset(height, width):
    """transform range (0, s) -> (-s/2,s/2)"""
    o_x = np.round(float(width) / 2)
    o_y = np.round(float(height) / 2)
    offset_matrix = np.array([[1, 0, o_x],
                              [0, 1, o_y],
                              [0, 0, 1]])
    return offset_matrix


def reset_offset(height, width):
    """transform range (-s/2,s/2) -> (0, s)"""
    o_x = np.round(float(width) / 2)
    o_y = np.round(float(height) / 2)
    reset_matrix = np.array([[1, 0, -o_x],
                             [0, 1, -o_y],
                             [0, 0, 1]])
    return reset_matrix


def random_rotate(rotation_range):
    """rotate around image center"""
    degree = np.random.uniform(-rotation_range / 2, rotation_range / 2)
    rotation_matrix = np.eye(3)
    rotation_matrix[:2] = cv2.getRotationMatrix2D((0, 0), degree, 1.0)
    return rotation_matrix


def random_translate(height_range, width_range):
    """translate around image"""
    tx = np.random.uniform(-height_range, height_range)
    ty = np.random.uniform(-width_range, width_range)
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    return translation_matrix


def random_shear(shear_range):
    """shear horizontally"""
    shear = np.random.uniform(-shear_range, shear_range)
    shear_matrix = np.array([[1, -math.sin(shear), 0],
                             [0, math.cos(shear), 0],
                             [0, 0, 1]])
    return shear_matrix


def random_zoom(zoom_range):
    """zoom or dezoom"""
    z = np.random.uniform(zoom_range[0], zoom_range[1])
    zoom_matrix = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]])
    return zoom_matrix


def random_horizontal_flip():
    """flip image"""
    if np.random.randint(2):
        mat = np.array([[-1, 0, -1],
                        [0, 1, 0],
                        [0, 0, 1]], dtype=np.float32)
    else:
        mat = np.eye(3)
    return mat


def get_affine_matrix(height, width):
    """get a random affine matrix 3x3 matrix"""
    mat = random_rotate(5)
    mat = random_translate(0.2 * height, 0.2 * width).dot(mat)
    mat = random_horizontal_flip().dot(mat)
    mat = mat.dot(random_zoom((0.75, 1.25)))
    return mat


def get_random_homography(height, width):
    """get a random homography 3x3 matrix"""
    mat = reset_offset(height, width)
    mat = get_affine_matrix(height, width).dot(mat)
    mat = offset(height, width).dot(mat)
    return mat




def angles_to_R(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = R_z.dot(R_y).dot(R_x)
    return R


def computeC2MC1(R1, tvec1, R2, tvec2):
    R_1to2 = R2.dot(R1.T)
    tvec_1to2 = R2.dot(-R1.T.dot(tvec1)) + tvec2
    return R_1to2, tvec_1to2


def computeHomography(R_1to2, tvec_1to2, dinv, normal):
    homography = R_1to2 - dinv * tvec_1to2.dot(normal)
    return homography


def compute_dinv(R1, tvec1, normal):
    origin = np.zeros((3), dtype=np.float32)
    origin1 = R1.dot(origin) + tvec1

    normal1 = R1.dot(normal)
    d_inv1 = 1.0 / normal1.dot(origin1)
    return d_inv1



if __name__ == '__main__':
    img = cv2.imread("datasets/scene.jpg", cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    K = np.array([[width/2, 0, width / 2],
                  [0, height/2, height / 2],
                  [0, 0, 1]], dtype=np.float32)
    Kinv = np.linalg.inv(K)

    rvec2 = np.zeros((3), dtype=np.float32)
    tvec2 = np.array([0,0,0], dtype=np.float32)

    R1 = np.eye(3)
    tvec1 = np.array([0,0,0], dtype=np.float32)
    normal = np.array([0,0,1], dtype=np.float32)

    alpha = 1e-6


    rvec_speed = np.random.randn(3) * 0.1
    tvec_speed = np.random.randn(3) * 0.1

    prev_img = img.astype(np.float32)

    t = 0

    while 1:
        rvec_speed = (1 - alpha) * rvec_speed + (alpha) * np.random.randn(3)
        tvec_speed = (1 - alpha) * tvec_speed + (alpha) * np.random.randn(3)


        rvec2 = rvec_speed * np.sin(t*0.01)
        tvec2 = tvec_speed * np.sin(t*0.01)


        R2 = angles_to_R(rvec2)
        R_1to2, tvec_1to2 = computeC2MC1(R1, tvec1, R2, tvec2)
        dinv = compute_dinv(R1, tvec1, normal)
        H = computeHomography(R_1to2, tvec_1to2, 1, normal)
        G = K.dot(H).dot(Kinv)
        G /= G[2,2]

        tx = np.sin(t*0.01) * width/2
        ty = np.cos(t * 0.01) * height/2
        z = np.cos(t*0.01)/2 + 1
        mat = reset_offset(height, width)
        mat = np.array([[z, 0, 0],
                            [0, z, 0],
                            [0, 0, 1]]).dot(mat)

        mat = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]]).dot(mat)

        G = offset(height, width).dot(mat).dot(G)


        out = cv2.warpPerspective(img, G, (width, height), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REFLECT101).astype(np.float32)
        out = (out-out.min())/(out.max()-out.min())

        diff = out-prev_img
        diff = (diff-diff.min())/(diff.max()-diff.min())

        cv2.imshow("diff", diff)
        cv2.imshow("out", out)
        cv2.waitKey(5)

        prev_img = out

        t += 1
