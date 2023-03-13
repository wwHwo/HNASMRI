import glob
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import h5py
import nibabel as nib
import numpy as np

# with open('/storage/c_HWW/NAS-MRI/dataProcessing/IXI.pickle', 'rb') as handle:
#     df_deepbrain_data = pkl.load(handle)
# files = glob.glob('/storage/c_HWW/dataset/IXI-T1/*.gz')
# name_slice = '/storage/c_HWW/dataset/IXI-T1/IXI188-Guys-0_150'
#
#
# brain_q = df_deepbrain_data[df_deepbrain_data['ID'] == name_slice].BRAIN_QUANTITY
f = open("/storage/c_HWW/dataset/IXI-T1/IXI641-Guys-1105-T1.nii.gz")
vol_np = nib.load(f).get_fdata()
print(vol_np.shape)
#img = vol_np[:,:,95]
#plt.figure()
#plt.subplot(1, 1, 1)
#plt.imshow(img, cmap=cm.gray)
