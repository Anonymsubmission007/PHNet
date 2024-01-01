import glob
import os
import nibabel as nib
import numpy as np


def align_shape(predict_folder, val_folder):  # align the shape in case the shape does not match
    pred_sort = sorted(glob.glob(os.path.join(predict_folder,"*nii.gz")))
    image_sort = sorted(glob.glob(os.path.join(val_folder,"*nii.gz")))
    for pred_path,image_path in zip(pred_sort,image_sort):
        pred = nib.load(pred_path)
        val_image = nib.load(image_path)
        prediction = np.asarray(pred.get_fdata())
        image = np.asarray(val_image.get_fdata())
        if prediction.shape != image.shape:  # the validation slices of this task are all (512,512)
            prediction = np.delete(prediction,512,axis=0)
            prediction = np.delete(prediction,512,axis=1)
            new_image = nib.Nifti1Image(prediction, val_image.affine) 
            nib.save(new_image,pred_path)
