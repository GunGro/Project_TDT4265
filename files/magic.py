import sys
import SimpleITK as sitk
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import PIL

def make_isotropic(image, interpolator = sitk.sitkLinear):
    '''
    Resample an image to isotropic pixels (using smallest spacing from original) and save to file. Many file formats
    (jpg, png,...) expect the pixels to be isotropic. By default the function uses a linear interpolator. For
    label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
    '''
    original_spacing = image.GetSpacing()
    # Image is already isotropic, just return a copy.
    if all(spc == original_spacing[0] for spc in original_spacing):
        return sitk.Image(image)
    # Make image isotropic via resampling.
    original_size = image.GetSize()
    min_spacing = min(original_spacing)
    new_spacing = [min_spacing]*image.GetDimension()
    new_size = [int(round(osz*ospc/min_spacing)) for osz,ospc in zip(original_size, original_spacing)]
    return sitk.Resample(image, new_size, sitk.Transform(), interpolator,
                         image.GetOrigin(), new_spacing, image.GetDirection(), 0,
                         image.GetPixelID())

def load_itk(filename, interpolator = sitk.sitkLinear):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    isotropic_image = make_isotropic(itkimage, interpolator)
    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

if __name__ == "__main__":
    Path_vec = sys.argv
    path = Path_vec[1]
    for path in Path_vec[1:-2]:
        # do ground_truth first
        for string in ["_2CH_ED_gt.mhd", "_4CH_ES_gt.mhd","_2CH_ES_gt.mhd", "_4CH_ED_gt.mhd"]:
            full_path = path + "/" + path + string
            arr, _, _ = load_itk(full_path, interpolator = sitk.sitkLinear)


            im = PIL.Image.fromarray(arr[0])
            im.save(Path_vec[-2] + "gt_" + path +string[:-7]+ ".tif")
        for string in ["_2CH_ED.mhd", "_4CH_ES.mhd","_2CH_ES.mhd", "_4CH_ED.mhd"]:
            full_path = path + "/" + path + string
            arr, _, _ = load_itk(full_path, interpolator = sitk.sitkNearestNeighbor)

            im = PIL.Image.fromarray(arr[0])
            im.save(Path_vec[-1] + "gray_" + path +string[:-4]+ ".tif")




        print(path)


