import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as sk


def build_gaussian_pyramid(image, level):
    """calcualtes the laplacian and gaussian pyramids with level >= 2"""
    if level < 2:
        exit(-1)
    gaussian_pyramid = [image]
    for _ in range(level - 1):  # You can adjust the number of levels in the pyramid
        blur = cv2.GaussianBlur(image,(5,5),0)
        image = cv2.pyrDown(blur)
        gaussian_pyramid.append(image)
    laplacian = []
    laplacian.append(gaussian_pyramid[-1])
    for i in range(level - 1, 0, -1):
        laplacian.append(gaussian_pyramid[i-1] - cv2.pyrUp(gaussian_pyramid[i]))
    return gaussian_pyramid, laplacian


def blend_images_with_mask(image1, image2, mask, levels):
    _, laplacian_pyr1 = build_gaussian_pyramid(image1,levels)
    _, laplacian_pyr2 = build_gaussian_pyramid(image2,levels)
    gaussian_pyr_mask, _ = build_gaussian_pyramid(mask, levels)

    blended_pyr = []
    for i in range(levels):
        blended_pyr.append(gaussian_pyr_mask[i] * laplacian_pyr1[levels-i-1] + (1 - gaussian_pyr_mask[i]) * laplacian_pyr2[levels-i - 1])

    blended_image = blended_pyr[-1]
    for i in range(len(blended_pyr) - 1, 1, -1):
        blended_image = cv2.pyrUp(blended_image) + blended_pyr[i - 1]

    return blended_image, laplacian_pyr2


def blend(image1_path, image2_path):
    image1 = sk.imread(image1_path).astype('float64')
    image2 = sk.imread(image2_path).astype('float64')
    mask = cv2.imread('mask.jpeg', 0)
    mask = np.clip(mask, 0, 1).astype('float64')
    red, green, blue = cv2.split(image1)
    result_red, lpr = blend_images_with_mask(image2[:, :, 0], image1[:, :, 0], mask, 9)
    result_green, lpg = blend_images_with_mask(image2[:, :, 1], image1[:, :, 1], mask, 9)
    result_blue, lpb = blend_images_with_mask(image2[:, :, 2], image1[:, :, 2], mask, 9)

    blended_image = np.stack([result_red, result_green, result_blue], axis=2)
    blended_image = np.array(blended_image).astype(int)
    return blended_image


def hypred(image1_path, image2_path, levels):
    image2 = sk.imread(image1_path).astype('float64')
    image1 = sk.imread(image2_path).astype('float64')
    _, laplacian_pyr1 = build_gaussian_pyramid(image1, levels)
    _, laplacian_pyr2 = build_gaussian_pyramid(image2, levels)
    blended_pyr = []
    for i in range(int(levels/2 + 1)):
        blended_pyr.append(laplacian_pyr1[i])
    for i in range(int(levels/2 + 1), levels):
        blended_pyr.append(laplacian_pyr2[i])

    blended_pyr.reverse()
    blended_image = blended_pyr[-1]

    for i in range(len(blended_pyr) - 1, 1, -1):
        blended_image = cv2.pyrUp(blended_image) + blended_pyr[i - 1]
    blended_image = np.array(blended_image).astype(int)
    return blended_image

# Example usage


# Ensure that the images and mask have the same dimensions
blended_image = hypred('/cs/usr/areen0507/Downloads/inshtain.jpeg','/cs/usr/areen0507/Downloads/marlyn.jpeg', 6)

plt.imshow(blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Plot original images
# plt.figure(figsize=(12, 4))
# plt.subplot(131), plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)), plt.title('Image A')
# plt.subplot(132), plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)), plt.title('Image B')
# plt.subplot(133), plt.imshow(mask, cmap='gray'), plt.title('Mask')
plt.show()
#
# # Plot Laplacian and Blended pyramids
# num_levels = len(lpr)
# plt.figure(figsize=(16, 8))
# for i in range(num_levels):
#     plt.imshow(lpr[i]*255)
# plt.show()