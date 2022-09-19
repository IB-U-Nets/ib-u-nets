from os import mkdir
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import argparse, os
from reformatMSDprostate import resample_3D_images

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, default="PROMISE_12", required=False, help="Folder/directory to read the original PROMISE-12 prostate data.")
    parser.add_argument("--out_folder", type=str, default="mri_framework/train_and_test", required=False, help="Folder/directory to save the converted data.")
    parser.add_argument("--change_spacing", action='store_true', help="If set, then data and corresponding label will be resampled to new_spacing.")
    parser.add_argument("--new_spacing", type=float, nargs=3, default=(0.613, 0.613, 3.6), required=False, help="Spacing to be resampled.")
    parser.add_argument("--change_direction", action='store_true', help="If set, then direction of data and corresponding label changed.")
    opt = parser.parse_args()
    
    in_folder = opt.in_folder
    out_folder = opt.out_folder

    # Create output folder.
    if not os.path.exists(os.path.join(out_folder)):
        os.makedirs(os.path.join(out_folder))

    # Get the paths of the data and labels.
    segmentations = subfiles(in_folder, suffix="segmentation.mhd")
    raw_data = [i for i in subfiles(in_folder, suffix="mhd") if not i.endswith("segmentation.mhd")]

    # Resample, change format to nrrd and save data and label in individually folders.
    if opt.change_spacing == True:
        for index, (dataPath, labelPath) in enumerate(zip(raw_data, segmentations)):
            mkdir(join(out_folder,str(index)))
            print("\nFolder number: " + str(index) + " " + dataPath + " " + labelPath)
            
            old_data = sitk.ReadImage(dataPath)
            new_data = resample_3D_images(sitkImage=old_data, newSpacing=opt.new_spacing,
                        interpolation="BSpline", change_spacing=opt.change_spacing, change_direction=opt.change_direction)
            data_fname = os.path.join(out_folder,str(index), "data.nrrd")
            sitk.WriteImage(sitk.Cast(new_data, sitk.sitkFloat32), data_fname, True)    
            
            old_label = sitk.ReadImage(labelPath)
            new_label = resample_3D_images(sitkImage=old_label, newSpacing=opt.new_spacing, 
                        interpolation="nearest", change_spacing=opt.change_spacing, change_direction=opt.change_direction)
            label_fname = os.path.join(out_folder,str(index), "label.nrrd")
            sitk.WriteImage(sitk.Cast(new_label, sitk.sitkUInt8), label_fname, True)

            print("The old data image has shape: " + str(old_data.GetSize()) + " with spacing: " + str(old_data.GetSpacing()))
            print("The new data image has shape: " + str(new_data.GetSize()) + " with spacing: " + str(new_data.GetSpacing()))
            print("The old label image has shape: " + str(old_label.GetSize()) + " with spacing: " + str(old_label.GetSpacing()))
            print("The new label image has shape: " + str(new_label.GetSize()) + " with spacing: " + str(new_label.GetSpacing()))

    #  No resampling, change format to nrrd and save data and label in individually folders.
    else:
        for index, (dataPath, labelPath) in enumerate(zip(raw_data, segmentations)):
            mkdir(join(out_folder,str(index)))
            print("Folder number: " + str(index) + " " + dataPath + " " + labelPath)
            
            data_fname = join(out_folder,str(index), "data.nrrd")
            sitk.WriteImage(sitk.ReadImage(dataPath), data_fname)
            
            label_fname = join(out_folder,str(index), "label.nrrd")
            sitk.WriteImage(sitk.ReadImage(labelPath), label_fname)
