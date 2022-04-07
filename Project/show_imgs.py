import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


directory = r'D:\OneDrive - University of Victoria\Files\Grad School\Classes\Machine Learning\OCT2017'
folder = 'train'  # train, test, or val

normal_path = os.path.join(directory, folder, 'NORMAL')
drusen_path = os.path.join(directory, folder, 'DRUSEN')
dme_path = os.path.join(directory, folder, 'DME')
cnv_path = os.path.join(directory, folder, 'CNV')

normal_files = [f for f in os.listdir(normal_path) if os.path.isfile(os.path.join(normal_path, f))]
drusen_files = [f for f in os.listdir(drusen_path) if os.path.isfile(os.path.join(drusen_path, f))]
dme_files = [f for f in os.listdir(dme_path) if os.path.isfile(os.path.join(dme_path, f))]
cnv_files = [f for f in os.listdir(cnv_path) if os.path.isfile(os.path.join(cnv_path, f))]


normal_img = np.array(Image.open(os.path.join(normal_path, normal_files[20])))
drusen_img = np.array(Image.open(os.path.join(drusen_path, drusen_files[21])))
dme_img = np.array(Image.open(os.path.join(dme_path, dme_files[23])))
cnv_img = np.array(Image.open(os.path.join(cnv_path, cnv_files[23])))

fig, ax = plt.subplots(1, 4)
ax[0].axis('off')
ax[1].axis('off')
ax[2].axis('off')
ax[3].axis('off')

ax[0].imshow(normal_img[0:376, 136:632], cmap='gray')
ax[0].set_title('NORMAL')
ax[1].imshow(drusen_img[40:416, 8:504], cmap='gray')
ax[1].set_title('DRUSEN')
ax[2].imshow(dme_img[30:406, 8:504], cmap='gray')
ax[2].set_title('DME')
ax[3].imshow(cnv_img[50:426, 8:504], cmap='gray')
ax[3].set_title('CNV')

plt.show()
