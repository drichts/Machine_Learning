#%% Randomly grab an equal number of images from the train, test, and validation files
# Copy them into a directory for collab
import os
import shutil
import numpy as np

directory = r'C:\Users\drich\Downloads\OCT2017'
save_dir = r'C:\Users\drich\Downloads\collab\OCT2017'

folders = ['train', 'test', 'val']
sub_folders = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
num_imgs = [1000, 100, 100]

for i, folder in enumerate(folders):
    img_per_sub = int(num_imgs[i] / 4)
    print(img_per_sub)
    for sub_folder in sub_folders:
        path = os.path.join(directory, folder, sub_folder)
        print(path)
        save_path = os.path.join(save_dir, folder, sub_folder)
        print(save_path)
        os.makedirs(save_path, exist_ok=True)

        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        files = np.array(files)
        np.random.shuffle(files)  # Shuffle the files

        if img_per_sub < len(files):
            actual_num_imgs = img_per_sub
        else:
            actual_num_imgs = len(files)
        print(actual_num_imgs)
        print()
        # Save the first img_per_sub imgs in the save location
        for n in np.arange(actual_num_imgs):
            loadfile = os.path.join(path, files[n])
            savefile = os.path.join(save_path, files[n])
            shutil.copyfile(loadfile, savefile)
