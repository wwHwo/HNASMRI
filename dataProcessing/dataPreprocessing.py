import glob
import pickle as pkl

import h5py
import nibabel as nib
import numpy as np

with open('/storage/c_HWW/NAS-MRI/dataProcessing/IXI.pickle', 'rb') as handle:
    df_deepbrain_data = pkl.load(handle)
# print(len(df_deepbrain_data[df_deepbrain_data.BRAIN_QUANTITY>=3000]))
# for i, row in df_deepbrain_data.iterrows():
#     df_deepbrain_data.at[i, 'ID'] = '/storage/'+df_deepbrain_data.at[i, 'ID']
# with open('deepbrain_image_data.pickle', 'wb') as handle:
#     pkl.dump(df_deepbrain_data, handle)
# print(df_deepbrain_data.sample(7))
save_path = "/storage/c_HWW/dataset/IXI-T1-h5/"
f = [i for i in range(1, 46860)]
arry = np.array(f)
np.random.shuffle(arry)
f_train = arry[0:23240]
f_val = arry[23240:32612]
f_test = arry[32612:]
files = glob.glob('/storage/c_HWW/dataset/IXI-T1/*.gz')
counts = 0
for f in files:
    vol_np = nib.load(f).get_fdata()
    name_vol = f[:-10]
    for idslice in range(vol_np.shape[2]):
        name_slice = name_vol + '_' + str(idslice)
        brain_q = df_deepbrain_data[df_deepbrain_data['ID'] == name_slice].BRAIN_QUANTITY
        if brain_q.values[0] > 9200:
            counts = counts + 1
            if counts in f_val:
                belong = 'val'
            elif counts in f_test:
                belong = 'test'
            elif counts in f_train:
                belong = 'train'
            else:
                continue
        else:
            continue
        img_slice = np.rot90(vol_np[:, :, idslice])
        assert (img_slice.shape == (256, 256))
        name_slice = name_slice.split('/')[5].split(".")[0]
        save = save_path + belong + '/' + name_slice + ".hdf5"
        with h5py.File(save, 'w') as w:
            w.create_dataset("data", data=img_slice)
print(f + " done")
