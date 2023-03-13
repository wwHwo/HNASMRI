import pickle as pkl
import numpy as np
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import h5py
import matplotlib.cm as cm

# with open('/storage/c_HWW/dataset/IXI-T1/deepbrain_image_data.pickle', 'rb') as handle:
#     df_deepbrain_data = pkl.load(handle)
# brain_q = int(df_deepbrain_data[df_deepbrain_data['ID'] == "/storage/c_HWW/dataset/IXI-T1/IXI188-Guys-0798-T1_141"]['BRAIN_QUANTITY'])
IXI = "/storage/c_HWW/dataset/IXI-T1-h5/val/IXI012-HH-1211_70.hdf5"
MICCAI = "/storage/c_HWW/dataset/MICCAI-h5/val/1000_3x1003_3_97.hdf5"
CC = "/storage/c_HWW/dataset/CC-h5/val/e13991s3_P0153649.hdf5"
f=h5py.File(IXI,"r")
data1 = f["data"]
f=h5py.File(MICCAI,"r")
data2 = f["data"]
f=h5py.File(CC,"r")
data3 = f["data"]
plt.subplot(1, 3, 1)
plt.imshow(data1, cmap=cm.gray)
plt.subplot(1, 3, 2)
plt.imshow(data2, cmap=cm.gray)
plt.subplot(1, 3, 3)
plt.imshow(data3, cmap=cm.gray)
plt.show()
