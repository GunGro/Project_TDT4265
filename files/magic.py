import sys
import SimpleITK as sitk
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import PIL
def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

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
            arr, _, _ = load_itk(full_path)

            im = PIL.Image.fromarray(arr[0])
            im.save(Path_vec[-2] + "gt_" + path +string[:-7]+ ".tif")
        for string in ["_2CH_ED.mhd", "_4CH_ES.mhd","_2CH_ES.mhd", "_4CH_ED.mhd"]:
            full_path = path + "/" + path + string
            arr, _, _ = load_itk(full_path)

            im = PIL.Image.fromarray(arr[0])
            im.save(Path_vec[-1] + "gray_" + path +string[:-4]+ ".tif")




        print(path)


