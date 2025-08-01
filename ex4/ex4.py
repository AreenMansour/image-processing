import cv2
import numpy as np


def ex4(p1, p2):

    # Read two images
    image1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

    mask = np.ceil(image1/255)
    image1 = cv2.GaussianBlur(image1, (11, 11), 0)

    # Feature extraction
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    # Feature matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # RANSAC
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Image warping
    height, width = image1.shape
    colored1 = cv2.imread(p1)
    colored2 = cv2.imread(p2)
    warped_image = cv2.warpPerspective(colored1, M, (width, height))

    mask = np.stack([mask, mask, mask], axis=2)
    mask = cv2.warpPerspective(mask, M, (width, height))

    result = (warped_image + (1-mask)*colored2)/255.0

    # Display results
    cv2.imshow('Image 1', colored1)
    cv2.imshow('Image 2', colored2)
    cv2.imshow('Warped Image 1', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
# high_res_part_path = '/cs/usr/areen0507/Downloads/desert_high_res.png'
# low_res_path = '/cs/usr/areen0507/Downloads/desert_low_res.jpg'

# ex4(high_res_part_path, low_res_path)
# blended_image = blend_images(low_res_path, high_res_part_path, window_size, k)
#
# plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB))
# plt.show()