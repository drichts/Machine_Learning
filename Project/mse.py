# Source: https://pyimagesearch.com/2014/09/15/python-compare-two-images/
import numpy as np


def mse(imageA, imageB):
    """
    The 'Mean Squared Error' between the two images is the sum of the squared difference between the two images.
    The lower the error, the more similar the two images are
    :param imageA: 2D ndarray
            The first image to compare. Both images must be the same dimensions
    :param imageB: 2D ndarray
            The second image to compare. Both images must be the same dimensions
    :return:
    """

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    return err
