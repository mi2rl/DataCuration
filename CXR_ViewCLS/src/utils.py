import SimpleITK as sitk
import numpy as np
from skimage import exposure 
from PIL import Image
import pandas as pd
from collections import OrderedDict

def hist_equalization( image, out_max=None):   
    return exposure.equalize_hist(image)#/out_max

def read_image(filePath):
    image = sitk.ReadImage(filePath)
    image = sitk.GetArrayFromImage(image).astype('float32')[0]

    return image

def load_image(path):
    img = read_image(path)
    shape = img.shape
    interpolation_size = abs(shape[0] - shape[1])

    copy_img = img.copy()

    copy_img -= np.min(copy_img)
    copy_img /= np.max(copy_img)
    copy_img *= 255.

    image = np.zeros((copy_img.shape[0], copy_img.shape[1], 3), dtype='uint8')
    image[:,:,0] = copy_img
    image[:,:,1] = copy_img
    image[:,:,2] = copy_img 

    image = np.asarray(Image.fromarray(image).resize((512, 512), Image.LINEAR))

    image = hist_equalization(image)
    image = np.expand_dims(image, axis=0)

    return image

def record_result():
    result_dic = OrderedDict()
    result_dic['file'] = []
    result_dic['path'] = []
    result_dic['pred'] = []
    
    return result_dic

def write_result(result, save_path):
    data_df = pd.DataFrame(result)
    data_df.to_csv(save_path)