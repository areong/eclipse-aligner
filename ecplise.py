import argparse
import os

import cv2 as cv
import numpy as np

def list_sorted_filenames(directory):
    """
    Get filenames in a directory, and return a sorted list of them.

    :param directory: The path to a directory
    :returns: A list of sorted filenames
    """
    with os.scandir(directory) as entries:
        filenames = [entry.name for entry in entries if entry.is_file()]
        filenames.sort()
        return filenames.copy()

def find_histogram_range(histogram):
    """
    Find the min and max index where histogram value is larger than zero.

    :param histogram: A histogram, e.g. an output of cv.calcHist().
    :returns: The min and max index with larger-than-zero value
    """
    size = len(histogram)
    min_i = 0
    while min_i < size:
        if histogram[min_i] > 0:
            break
        min_i += 1

    max_i = size - 1
    while max_i >= 0:
        if histogram[max_i] > 0:
            break
        max_i -= 1
    return min_i, max_i

def find_largest_enclosing_circle(img):
    """
    Find the largest enclosing circle.
    Enclosing circles are found by converting the input grayscale image
    to a binary image at the middle intensity.

    :param img: An unsigned 8-bit grayscale image
    :returns: The center and radius of the largest enclosing circle.
        If no contour is found, return zero values.
    :raise ValueError: Raise at invalid input.
    """
    if img.dtype is not np.dtype(np.uint8):
        raise ValueError('The input image data type should be uint8.')

    # Calculate histogram.
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    # Find the min and max intensity value on the image.
    min_i, max_i = find_histogram_range(hist)

    # Threshold the image at the median intensity.
    _, binary_img = cv.threshold(img, (max_i + min_i) / 2, 255, cv.THRESH_BINARY)

    # Find contours.
    contours, _ = cv.findContours(binary_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    if len(contours) == 0:
        return (0, 0), 0

    # Find a minimum enclosing circle for each contour, and find the largest one.
    circles = [cv.minEnclosingCircle(contour) for contour in contours]
    max_circle = max(circles, key=lambda circle: circle[1])
    (center_x, center_y), radius = max_circle
    return (int(center_x), int(center_y)), int(radius)

def normalize_image(img, circle_center, circle_radius, target_circle_radius):
    """
    Transform an image such that the provided circle is moved to the image center
    and scaled to the specified size.

    :param img: An image
    :param circle_center: Circle center
    :param circle_radius: Circle radius
    :param target_circle_radius: Target circle radius
    :returns: A transformed image
    """
    # Center the image to the circle center.
    num_rows, num_cols, _ = img.shape
    img_center_x = num_cols / 2
    img_center_y = num_rows / 2
    scale = target_circle_radius / circle_radius

    # Create translation and scaling matrices.
    origin_to_circle_center_translation = np.float32(
        [[1, 0, -1 * circle_center[0]],
         [0, 1, -1 * circle_center[1]],
         [0, 0, 1]])
    circle_centered_scaling = np.float32(
        [[scale, 0, 0],
         [0, scale, 0],
         [0, 0, 1]])
    circle_center_to_new_origin_translation = np.float32(
        [[1, 0, img_center_x],
         [0, 1, img_center_y],
         [0, 0, 1]])
    transformation_matrix = np.matmul(
        circle_center_to_new_origin_translation,
        np.matmul(circle_centered_scaling, origin_to_circle_center_translation))

    # Apply the transformation.
    # Note the x-y order of image size.
    return cv.warpAffine(img, transformation_matrix[:2], (num_cols, num_rows))

def main(directory):
    """
    Align and show images.

    :param directory: A path of directory where images are located
    """
    # List sorted filenames under a directory.
    # Remove ending '/' if any.
    directory.rstrip('/')
    filenames = list_sorted_filenames(directory)
    print(len(filenames))

    target_circle_radius = 1000
    target_circle_radius_is_set = False

    cv.namedWindow('image', cv.WINDOW_NORMAL)

    for filename in filenames:
        # Read an image.
        img = cv.imread(directory + '/' + filename, cv.IMREAD_COLOR)
        grayscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the largest enclosing circle.
        circle_center, circle_radius = find_largest_enclosing_circle(grayscale_img)
        if circle_radius == 0:
            # No circle is found. Skip.
            continue
        cv.circle(img, circle_center, circle_radius, (0, 255, 0), 2)

        # Use the radius of the first image as the target radius.
        if not target_circle_radius_is_set:
            target_circle_radius = circle_radius
            target_circle_radius_is_set = True

        # Center the image to the circle center, and scale to the target circle radius.
        aligned_img = normalize_image(img, circle_center, circle_radius, target_circle_radius)

        cv.imshow('image', aligned_img)
        cv.waitKey(0)

        # Allow the user to interactively adjust the found circle.

        # Allow the user to reset the image transformation.

        # Allow the user to mark an image to be disgarded.

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Align eclipse photos.')
    argparser.add_argument('dir', type=str, help='A directory where images are located.')
    args = argparser.parse_args()
    main(args.dir)
