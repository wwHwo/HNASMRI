import nibabel as nib
import os
import glob
import pandas as pd
import numpy as np
import pickle as pkl
from deepbrain import Extractor
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
df_deepbrain_data = pd.DataFrame(columns=['ID', 'BRAIN_QUANTITY'])
files = glob.glob('/storage/c_HWW/dataset/IXI-T1/*.gz')
ext = Extractor()
for f in files:

    # get nibabel file
    vol_np = nib.load(f).get_fdata()

    # get brain mask
    prob = ext.run(vol_np)
    mask = prob > 0.5
    for id_sag_slice in range(vol_np.shape[2]):
        # Append data
        df_deepbrain_data = df_deepbrain_data.append({'ID': f[:-13] + '_' + str(id_sag_slice),
                                                      'BRAIN_QUANTITY': np.sum(mask[:, :, id_sag_slice])
                                                      }, ignore_index=True)

df_deepbrain_data = df_deepbrain_data.astype({'BRAIN_QUANTITY': 'float64'})

with open('IXI.pickle', 'wb') as handle:
    pkl.dump(df_deepbrain_data, handle)
#
# with open('deepbrain_image_data.pickle', 'rb') as handle:
#     df_deepbrain_data = pkl.load(handle)
