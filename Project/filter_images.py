import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from crimmins import crimmins
import image_filters as imf
from datetime import datetime
from scipy.ndimage import gaussian_filter
from scipy.signal import wiener
from compare_images import compare_images

directory = r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\OCT2017'
folder = 'train'  # train, test, or val
group = 'NORMAL'  # CNV, DME, DRUSEN, NORMAL

path = os.path.join(directory, folder, group)

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

img = np.load('image.npy')

# img = Image.open(os.path.join(path, files[0]))
# img = np.array(img)
# np.save('image.npy', img)
#
# start = datetime.now().timestamp()
# kuan_img = imf.kuan_filter(img)
# stop = datetime.now().timestamp()
# print(f'Kuan: {stop-start} s')
# np.save('kuan_img.npy', kuan_img)
#
# start = datetime.now().timestamp()
# frost_img = imf.frost_filter(img)
# stop = datetime.now().timestamp()
# print(f'Frost: {stop-start} s')
# np.save('frost_img.npy', frost_img)
#
# start = datetime.now().timestamp()
# lee_img = imf.lee_filter(img, 3)
# stop = datetime.now().timestamp()
# print(f'Lee: {stop-start} s')
# np.save('lee_img.npy', lee_img)
#
# start = datetime.now().timestamp()
# median_img = median_filter(img, size=3)
# stop = datetime.now().timestamp()
# print(f'Median: {stop-start} s')
# np.save('median_img.npy', median_img)

# start = datetime.now().timestamp()
# conv_img = imf.conservative_filter(img, 3)
# stop = datetime.now().timestamp()
# print(f'Conservative: {stop-start} s')
# np.save('conv_img.npy', conv_img)

# start = datetime.now().timestamp()
# crim_img = crimmins(img)
# stop = datetime.now().timestamp()
# print(f'Crimmins: {stop-start} s')
# np.save('crim_img.npy', crim_img)

# start = datetime.now().timestamp()
# gaus_img = gaussian_filter(img, sigma=2)
# stop = datetime.now().timestamp()
# print(f'Gaussian: {stop-start} s')
# np.save('gaus_img.npy', gaus_img)

# start = datetime.now().timestamp()
# wien_img = wiener(img, 5)
# stop = datetime.now().timestamp()
# print(f'Wiener: {stop-start} s')
# np.save('wien_img.npy', wien_img)


conv_img = np.load('conv_img.npy')
lee_img = np.load('lee_img.npy')
median_img = np.load('median_img.npy')
wien_img = np.load('wien_img.npy')
gaus_img = np.load('gaus_img.npy')


print(f'Conservative: {compare_images(img, conv_img)}')
print(f'Lee: {compare_images(img, lee_img)}')
print(f'Median: {compare_images(img, median_img)}')
print(f'Weiner: {compare_images(img, wien_img)}')
print(f'Gaussian: {compare_images(img, gaus_img)}')


fig, ax = plt.subplots(6, 1)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[3].axis('off')
ax[4].axis('off')
ax[5].axis('off')

ax[0].imshow(img, cmap='gray', interpolation='nearest')
ax[0].set_title('Unfiltered')

ax[1].imshow(np.uint8(lee_img), cmap='gray')
ax[1].set_title('Lee')

ax[2].imshow(np.uint8(conv_img), cmap='gray')
ax[2].set_title('Conservative')

ax[3].imshow(np.uint8(median_img), cmap='gray')
ax[3].set_title('Median')

ax[4].imshow(np.uint8(wien_img), cmap='gray')
ax[4].set_title('Wiener')

ax[5].imshow(np.uint8(gaus_img), cmap='gray')
ax[5].set_title('Gaussian')

plt.show()
