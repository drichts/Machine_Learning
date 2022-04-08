from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance


def lee_filter(img, size):
    """
    Apply a Lee filter to your image
    https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    :param img: 2D ndarray
            The image to filter
    :param size: int
            The size of the filter, i.e. 3 is a 3x3 window
    :return:
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)

    return img_output


def conservative_filter(data, filter_size):
    """
    Apply conservative smoothing to correct salt and pepper noise
    https://towardsdatascience.com/image-filters-in-python-26ee938e57d2
    :param data: 2D ndarray
            The image to filter
    :param filter_size: int
            The size of the filter, i.e. 3 is a 3x3 window
    :return:
    """
    temp = []
    indexer = filter_size // 2
    new_image = data.copy()
    nrow, ncol = data.shape

    for i in range(nrow):
        for j in range(ncol):
            for k in range(i - indexer, i + indexer + 1):
                for m in range(j - indexer, j + indexer + 1):
                    if (k > -1) and (k < nrow):
                        if (m > -1) and (m < ncol):
                            temp.append(data[k, m])

            temp.remove(data[i, j])
            max_value = max(temp)
            min_value = min(temp)

            if data[i, j] > max_value:
                new_image[i, j] = max_value
            elif data[i, j] < min_value:
                new_image[i, j] = min_value

            temp = []

    return new_image.copy()
