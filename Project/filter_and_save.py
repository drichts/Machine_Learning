#%% Randomly grab an equal number of images from the train, test, and validation files
# Copy them into a directory for collab
import os
import numpy as np
from image_filters import lee_filter, conservative_filter
from PIL import Image
from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter


directory = r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\OCT2017'
save_dir = r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\OCT2017_gaus'

folders = ['train', 'test', 'val']
sub_folders = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

for i, folder in enumerate(folders):

    for sub_folder in sub_folders:
        path = os.path.join(directory, folder, sub_folder)
        save_path = os.path.join(save_dir, folder, sub_folder)
        os.makedirs(save_path, exist_ok=True)

        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files = np.array(files)

        # Save the first img_per_sub imgs in the save location
        for file in files:
            loadfile = os.path.join(path, file)
            savefile = os.path.join(save_path, file)
            # print(loadfile)
            # print(savefile)
            # print()
            img = Image.open(os.path.join(path, file))
            img = np.array(img)
            # img = lee_filter(img, 3)
            # img = median_filter(img, size=3)
            # img = conservative_filter(img, 3)
            img = gaussian_filter(img, sigma=2)
            img = Image.fromarray(np.uint8(img))
            img.save(savefile)
