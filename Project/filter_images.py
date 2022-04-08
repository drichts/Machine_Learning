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
from mse import compare_images

directory = r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\OCT2017'
folder = 'train'  # train, test, or val
group = 'DME'  # CNV, DME, DRUSEN, NORMAL

path = os.path.join(directory, folder, group)

files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

# img = np.load('image.npy')

img = Image.open(os.path.join(path, files[11]))
img = np.array(img)[0:-100]
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
start = datetime.now().timestamp()
lee_img = imf.lee_filter(img, 3)
stop = datetime.now().timestamp()
print(f'Lee: {stop-start} s')
# np.save('lee_img.npy', lee_img)
#
start = datetime.now().timestamp()
median_img = median_filter(img, size=3)
stop = datetime.now().timestamp()
print(f'Median: {stop-start} s')
# np.save('median_img.npy', median_img)

start = datetime.now().timestamp()
conv_img = imf.conservative_filter(img, 3)
stop = datetime.now().timestamp()
print(f'Conservative: {stop-start} s')
# np.save('conv_img.npy', conv_img)

# start = datetime.now().timestamp()
# crim_img = crimmins(img)
# stop = datetime.now().timestamp()
# print(f'Crimmins: {stop-start} s')
# np.save('crim_img.npy', crim_img)

start = datetime.now().timestamp()
gaus_img = gaussian_filter(img, sigma=2)
stop = datetime.now().timestamp()
print(f'Gaussian: {stop-start} s')
# np.save('gaus_img.npy', gaus_img)

# start = datetime.now().timestamp()
# wien_img = wiener(img, 5)
# stop = datetime.now().timestamp()
# print(f'Wiener: {stop-start} s')
# np.save('wien_img.npy', wien_img)

#
# conv_img = np.load('conv_img.npy')
# lee_img = np.load('lee_img.npy')
# median_img = np.load('median_img.npy')
# # wien_img = np.load('wien_img.npy')
# gaus_img = np.load('gaus_img.npy')


print(f'Conservative: {compare_images(img, conv_img)}')
print(f'Lee: {compare_images(img, lee_img)}')
print(f'Median: {compare_images(img, median_img)}')
# print(f'Weiner: {compare_images(img, wien_img)}')
print(f'Gaussian: {compare_images(img, gaus_img)}')


fig1 = plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.title('Unfiltered')
plt.show()
fig1.savefig(r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\filtered_img1.png', dpi=500)

fig2 = plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(np.uint8(lee_img), cmap='gray')
plt.title('Lee')
plt.show()
fig2.savefig(r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\filtered_img2.png', dpi=500)

fig3 = plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(np.uint8(conv_img), cmap='gray')
plt.title('Conservative')
plt.show()
fig3.savefig(r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\filtered_img3.png', dpi=500)

fig4 = plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(np.uint8(median_img), cmap='gray')
plt.title('Median')
plt.show()
fig4.savefig(r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\filtered_img4.png', dpi=500)

fig5 = plt.figure(figsize=(5, 5))
plt.axis('off')
plt.imshow(np.uint8(gaus_img), cmap='gray')
plt.title('Gaussian')
plt.show()
fig5.savefig(r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\filtered_img5.png', dpi=500)
