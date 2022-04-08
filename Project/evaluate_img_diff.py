import os
import numpy as np
from image_filters import lee_filter, conservative_filter
from PIL import Image
from scipy.ndimage import median_filter
from mse import compare_images

directory = r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning'

folder = 'train'
sub_folders = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

scores = np.zeros((4, 1000, 2))

for sub_folder in sub_folders:
    path = os.path.join(directory, 'OCT2017', folder, sub_folder)
    conv_path = os.path.join(directory, 'OCT2017_conv', folder, sub_folder)
    lee_path = os.path.join(directory, 'OCT2017_Lee', folder, sub_folder)
    med_path = os.path.join(directory, 'OCT2017_med', folder, sub_folder)
    gaus_path = os.path.join(directory, 'OCT2017_gaus', folder, sub_folder)

    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    files = np.array(files)

    for idx, file in enumerate(files):
        loadfile = os.path.join(path, file)
        convfile = os.path.join(conv_path, file)
        leefile = os.path.join(lee_path, file)
        medfile = os.path.join(med_path, file)
        gausfile = os.path.join(gaus_path, file)

        img = Image.open(loadfile)
        img = np.array(img)

        convimg = Image.open(convfile)
        convimg = np.array(convimg)

        leeimg = Image.open(leefile)
        leeimg = np.array(leeimg)

        medimg = Image.open(medfile)
        medimg = np.array(medimg)

        gausimg = Image.open(gausfile)
        gausimg = np.array(gausimg)

        scores[0, idx] = compare_images(img, convimg)
        scores[1, idx] = compare_images(img, medimg)
        scores[2, idx] = compare_images(img, leeimg)
        scores[3, idx] = compare_images(img, gausimg)

mean_scores = np.mean(scores, axis=1)
std_scores = np.std(scores, axis=1)

print(mean_scores)
print(std_scores)
