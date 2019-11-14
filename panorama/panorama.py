# -*- coding: utf-8 -*-
# @Time    : 2019/11/13 16:21
# @Author  : 0chen
# @FileName: panorama.py
# @Software: PyCharm
# @Blog    : http://www.0chen.xyz

import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def ransc(kp1, kp2, t, deviation):
    kp1_add_one = np.hstack((kp1, np.ones((kp1.shape[0], 1))))
    index = range(len(kp1))
    max_correct, ans_index = 0, []
    while t:
        t -= 1
        selected = random.sample(index, 4)
        homo, _ = cv2.findHomography(kp1[selected], kp2[selected])

        dst = np.dot(kp1_add_one, homo.T)
        dst = dst/(dst[:, -1].reshape(dst.shape[0], 1))
        dst = dst[:, :2]

        correct, ans = 0, []
        for i in index:
            if np.linalg.norm(dst[i] - kp2[i]) < deviation:
                correct += 1
                ans.append(i)

        if max_correct <= correct:
            max_correct = correct
            ans_index = ans

    return ans_index

def show(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def match_point(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    return good

if __name__ == '__main__':
    img1 = cv2.imread('picture/1.jpg', cv2.IMREAD_COLOR)
    img2 = cv2.imread('picture/2.jpg', cv2.IMREAD_COLOR)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, descriptors1 = sift.detectAndCompute(img1, None)
    kp2, descriptors2 = sift.detectAndCompute(img2, None)
    h, w, z = img1.shape

    matches = match_point(descriptors1, descriptors2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show(img3)

    fir_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1,  2)
    sec_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1,  2)
    sec_pts[:, 0] += w

    index = ransc(fir_pts, sec_pts, 100, 10)

    img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, np.array(matches)[index], None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    show(img4)

    homography, _ = cv2.findHomography(fir_pts[index], sec_pts[index])
    img5 = cv2.warpPerspective(img1, homography, (w*2, h))
    img5[0:h, w:w*2] = img2
    show(img5)

