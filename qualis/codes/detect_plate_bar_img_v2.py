
import argparse

import cv2
import numpy as np

import matplotlib.pyplot as plt

def get_corrected_img(img1, img2):
    MIN_MATCHES = 50

    # Display traning image and testing image
    fx, plots = plt.subplots(1, 2, figsize=(20,10))

    plots[0].set_title("Training Image")
    plots[0].imshow(img2)

    plots[1].set_title("Testing Image")
    plots[1].imshow(img1)
    plt.show()

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        corrected_img = cv2.warpPerspective(img1, m, (img2.shape[1], img2.shape[0]))

        return corrected_img
    return img2


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument("--src", default='../imgs/in/second_tests/bar_reference_noir.jpg', help="path for the object image")
    # parser.add_argument("--dest", default='../imgs/in/second_tests/image_2021-08-09_10:02:35.jpg', help="path for image containing the object")
    # args = parser.parse_args()

    im1 = cv2.imread('../imgs/in/second_tests/bar_reference2.jpg')
    im2 = cv2.imread('../imgs/in/second_tests/image_2021-08-09_10:02:35.jpg')

    img = get_corrected_img(im2, im1)
    # cv2.imshow('Corrected image', img)
    # cv2.waitKey()

    plt.imshow(img)
    plt.show()