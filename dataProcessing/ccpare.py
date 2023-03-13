import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.fftpack
import pickle
import glob
import h5py
import random
import shutil

data = np.load("/storage/c_HWW/dataset/Single-channel/Train_part1/Train/e13991s3_P01536.7.npy")
x = data[99,:,:,:]
x = x[:,:,0] + 1j*x[:,:,1]
# print(x.shape)
# print(x.dtype)
x = np.fft.ifftshift(x)
x = np.fft.ifft2(x)
x = np.abs(x)
plt.figure()
plt.imshow(x, cmap=cm.gray)
plt.show()
# mask = open(
#         '/storage/c_HWW/Subsampled-Brain-MRI-Reconstruction-by-Generative-Adversarial-Neural-Networks-master/Masks/mask_30_256.pickle',
#         'rb')
# mask = pickle.load(mask)
# mask = mask['mask0']
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(x, cmap=cm.gray)
# # x = data[99,:,:,:]
# # x = x[:,:,0] + 1j*x[:,:,1]
# x= np.fft.fft2(x)
# x = np.fft.fftshift(x)
# x = x * mask
# x = np.fft.ifftshift(x)
# x = np.fft.ifft2(x)
# x = np.abs(x)
# plt.subplot(1, 2, 2)
# plt.imshow(x, cmap=cm.gray)
# #plt.get_current_fig_manager().window.showMaximized()
# plt.show()
# file = glob.glob('/storage/c_HWW/dataset/Single-channel/Val/*.npy')
# save_path = '/storage/c_HWW/dataset/CC-h5/'
# for f in file:
#     data = np.load(f)
#     for i in range(data.shape[0]):
#         x = data[i]
#         x = x[:, :, 0] + 1j * x[:, :, 1]
#         x = np.fft.ifftshift(x)
#         x = np.fft.ifft2(x)
#         x = np.abs(x)
#         save = save_path + f.split('/')[6].split(".")[0] + str(i) + ".hdf5"
#         with h5py.File(save, 'w') as w:
#             w.create_dataset("data", data=x)
#
#     print("%s done" %f)
# file = glob.glob('/storage/c_HWW/dataset/CC-h5/*.hdf5')
# sample = random.sample(file,1000)
# for i in sample:
#     shutil.move(i,'/storage/c_HWW/dataset/CC-h5/test')
