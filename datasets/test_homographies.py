from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2

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


if __name__ == '__main__':
    img = cv2.imread("datasets/scene.jpg", cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    K = np.array([[width / 2, 0, width / 2],
                  [0, width/2, height / 2],
                  [0, 0, 1]], dtype=np.float32)
    Kinv = np.linalg.inv(K)

    rvec1 = np.array([0,0,0], dtype=np.float32)
    tvec1 = np.array([0,0,0], dtype=np.float32)
    nt = np.array([0,0,-1], dtype=np.float32).reshape(1,3)



    rvec_speed = np.random.rand(3) * 0.25
    tvec_speed = np.random.rand(3) * 0.5
    tshift = np.random.randn(3)

    prev_img = img.astype(np.float32)

    d = 1
    t = 0

    while 1:
        tshift = moving_average(tshift, 1e-4)


        rvec2 = rvec_speed * np.sin(t*0.01)
        tvec2 = tvec_speed * np.sin(t*0.01 + tshift)


        G_0to2 = generate_homography(rvec1, tvec1, rvec2, tvec2, nt, K, Kinv, d)



        out = cv2.warpPerspective(img, G_0to2, (width, height),
                                  flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        out = (out-out.min())/(out.max()-out.min())

        diff = out-prev_img
        diff = (diff-diff.mean())/(1e-3 + diff.std())
        diff = (diff-diff.min())/(diff.max()-diff.min())

        cv2.imshow("diff", diff)
        cv2.imshow("out", out)
        cv2.waitKey(5)

        prev_img = out

        t += 1
