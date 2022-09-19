'''
This script converts the Prostate Segmentation MRI data and ground-truth labels to the required format.
'''

from os import mkdir
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np
import argparse
import os

def getImgExtent(sitkImage, newImage):
  imgSize =sitkImage.GetSize()
  xExtent = [0, imgSize[0]]
  yExtent = [0, imgSize[1]]
  zExtent = [0, imgSize[2]]
  
  minValues = [float('inf'),float('inf'),float('inf')]
  maxValues = [float('-inf'),float('-inf'),float('-inf')]
  
  for xVal in xExtent:
    for yVal in yExtent:
      for zVal in zExtent:
        worldCoordinates = sitkImage.TransformIndexToPhysicalPoint((xVal,yVal,zVal))
        idxCoordinates = newImage.TransformPhysicalPointToIndex(worldCoordinates)
        for idx in range(0,len(minValues)):
          if idxCoordinates[idx] < minValues[idx]:
            minValues[idx] = idxCoordinates[idx]
          if idxCoordinates[idx] > maxValues[idx]:
            maxValues[idx] = idxCoordinates[idx]

  minWorldValues = newImage.TransformIndexToPhysicalPoint(minValues)
  voxelExtent = np.subtract(maxValues,minValues)
  return minWorldValues,voxelExtent

'''
Functions "get3dslice" and parts of "resample_4D_images" are from https://discourse.itk.org/t/resampleimagefilter-4d-images/2172/2
Author: SachidanandAlle
Date Taken: 2 July 2021
'''
def get3dslice(image, slice=0):
    size = list(image.GetSize())
    if len(size) == 4:
        size[3] = 0
        index = [0, 0, 0, slice]

        extractor = sitk.ExtractImageFilter()
        extractor.SetSize(size)
        extractor.SetIndex(index)
        image = extractor.Execute(image)
    return image

def resample_4D_images(sitkImage, newSpacing, interpolation="trilinear", 
                        newDirection=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) ,change_spacing=False, change_direction=False):
    """
    input image will be resampled.
    """    
    # Resample 4D (SITK Doesn't support directly; so iterate through slice and get it done)
    #new_data_list = []
    size = list(sitkImage.GetSize())
    for s in range(size[3]):
        img = get3dslice(sitkImage, s)
        img = resample_3D_images(sitkImage=img, newSpacing=newSpacing, interpolation=interpolation, newDirection=newDirection, change_spacing=change_spacing, change_direction=change_direction )
        #new_data_list.append(img)
        break # Get only the first slice T2 modality. 

    #joinImages = sitk.JoinSeriesImageFilter()
    #newimage = joinImages.Execute(new_data_list)
    newimage = img
    return newimage

def resample_3D_images(sitkImage, newSpacing, interpolation="trilinear", 
                        newDirection=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) ,change_spacing=False, change_direction=False):
    """
    input image will be resampled.
    """    

    resImgFiler = sitk.ResampleImageFilter()
    if change_spacing:
        resImgFiler.SetOutputSpacing(newSpacing)
    else:
        resImgFiler.SetOutputSpacing(sitkImage.GetSpacing())
    
    if interpolation == "BSpline":
        resImgFiler.SetInterpolator(sitk.sitkBSpline)
    elif interpolation == "nearest":
        resImgFiler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolation == "trilinear":
        resImgFiler.SetInterpolator(sitk.sitkLinear)

    resampledImage = sitk.Image(5, 5, 5, sitk.sitkInt16)
    if change_direction:
        resampledImage.SetDirection(newDirection)
    else:
        resampledImage.SetDirection(sitkImage.GetDirection())

    resampledImage.SetOrigin(sitkImage.GetOrigin())
    if change_spacing:
        resampledImage.SetSpacing(newSpacing)
    else:
        resampledImage.SetSpacing(sitkImage.GetSpacing())
            
    [newOrigin, newSize]= getImgExtent(sitkImage, resampledImage)
    # Ensuring a minimum size of 16 is present in the last dimension.
    if newSize[-1] < 16:
        newSize[-1] = 17
    resImgFiler.SetSize(sitk.VectorUInt32(newSize.tolist()))
    resImgFiler.SetOutputOrigin( newOrigin )
    if change_direction:
        resImgFiler.SetOutputDirection(newDirection )
    else:
        resImgFiler.SetOutputDirection(sitkImage.GetDirection())

    trans=sitk.Transform()
    trans.SetIdentity()
    resImgFiler.SetTransform( trans ) 
    resampledImage = resImgFiler.Execute(sitkImage)

    return resampledImage


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_folder", type=str, default="Task05_Prostate", required=False, help="Folder/directory to read the original MSD prostate data.")
    parser.add_argument("--out_folder", type=str, default="mri_framework/train_and_test", required=False, help="Folder/directory to save the converted data.")
    parser.add_argument("--change_spacing", action='store_true', help="If set, then data and corresponding label will be resampled to new_spacing.")
    parser.add_argument("--new_spacing", type=float, nargs=3, default=(0.625, 0.625, 3.6), required=False, help="Spacing to be resampled.")
    parser.add_argument("--change_direction", action='store_true', help="If set, then direction of data and corresponding label changed.")
    opt = parser.parse_args()
    
    in_folder = opt.in_folder
    out_folder = opt.out_folder

    # Create output folder.
    if not os.path.exists(os.path.join(out_folder)):
        os.makedirs(os.path.join(out_folder))

    # Get the paths of the data and labels.
    raw_data = subfiles(join(in_folder,"imagesTr"), suffix=".nii.gz")
    segmentations = subfiles(join(in_folder,"labelsTr"), suffix=".nii.gz")

    # Resample, change format to nrrd and save data and label in individually folders.
    if opt.change_spacing == True:
        for index, (dataPath, labelPath) in enumerate(zip(raw_data, segmentations)):
            mkdir(join(out_folder,str(index)))
            print("\nFolder number: " + str(index) + " " + dataPath + " " + labelPath)
            
            old_data = sitk.ReadImage(dataPath)
            new_data = resample_4D_images(sitkImage=old_data, newSpacing=opt.new_spacing,
                        interpolation="BSpline", change_spacing=opt.change_spacing, change_direction=opt.change_direction)
            data_fname = os.path.join(out_folder,str(index), "data.nrrd")
            sitk.WriteImage(sitk.Cast(new_data, sitk.sitkFloat32), data_fname, True)    
            
            old_label = sitk.ReadImage(labelPath)
            new_label = resample_3D_images(sitkImage=old_label, newSpacing=opt.new_spacing, 
                        interpolation="nearest", change_spacing=opt.change_spacing, change_direction=opt.change_direction)
            label_fname = os.path.join(out_folder,str(index), "label.nrrd")
            sitk.WriteImage(sitk.Cast(new_label, sitk.sitkFloat32), label_fname, True)

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
