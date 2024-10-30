import SimpleITK as sitk
import nibabel as nib
import os
import pydicom
import numpy as np

def read_dicom_and_convert_to_numpy(dicom_path):
    # DICOM 파일 읽기
    dicom_file = pydicom.dcmread(dicom_path, force=True)

    # 이미지 데이터를 NumPy 배열로 변환
    image_array = dicom_file.pixel_array

    # Rescale Slope와 Rescale Intercept 적용
    rescale_slope = dicom_file.RescaleSlope if 'RescaleSlope' in dicom_file else 1
    rescale_intercept = dicom_file.RescaleIntercept if 'RescaleIntercept' in dicom_file else 0
    image_array = image_array * rescale_slope + rescale_intercept

    return image_array.astype(np.float32)[None]

def read_dicom_series_to_array(dicom_path_list):
    try:
        # DICOM 시리즈를 읽습니다.
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_path_list)
        image = reader.Execute()
        # 이미지를 NumPy array로 변환합니다.
        array = sitk.GetArrayFromImage(image)
    except:
        try:
            array = [(int(pydicom.dcmread(dcm_path, force=True)["InstanceNumber"].value), 
                      read_dicom_and_convert_to_numpy(dcm_path))
                     for dcm_path in dicom_path_list]
        except:
            try:
                array = [(dcm_path, read_dicom_and_convert_to_numpy(dcm_path))
                         for dcm_path in dicom_path_list]
            except:
                return None
        array = sorted(array, key=lambda x: x[0])
        array = np.concatenate([item[1] for item in array], axis=0)
    return array

def get_dicom_series_shape(dicom_path_list):
    # 첫 번째 DICOM 파일을 읽어서 한 이미지의 크기를 얻습니다.
    single_image = sitk.ReadImage(dicom_path_list[0])
    single_image_size = single_image.GetSize()

    # 시리즈의 전체 차원을 계산합니다 (가정: 모든 이미지가 같은 크기).
    series_shape = (len(dicom_path_list), single_image_size[1], single_image_size[0])
    return series_shape

def read_nii_to_array(nii_path):
    # NIfTI 파일을 읽습니다.
    image = sitk.ReadImage(nii_path)

    # 이미지를 NumPy array로 변환합니다.
    array = sitk.GetArrayFromImage(image)
    # 이미지의 Affine 행렬을 가져옵니다.
    affine_matrix = np.array(image.GetDirection()).reshape(3, 3)

#     # 뒤집힘 여부를 결정합니다.
#     reverse = np.linalg.det(affine_matrix) > 0
#     if reverse:
#         array = array[::-1]
    return array

def get_reversed_mask(mask_array):
    return mask_array[::-1]

def get_nii_shape(nii_path):
    # NIfTI 파일의 헤더를 읽습니다.
    nii_header = nib.load(nii_path).header

    # 차원 정보를 얻습니다.
    nii_shape = nii_header.get_data_shape()
    return (nii_shape[2], nii_shape[1], nii_shape[0])

def resize_dicom_series(image, resize_factor_list, nearest=False):

    dimension = image.GetDimension()

    reference_physical_size = np.zeros(image.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(image.GetSize(), 
                                                                                        image.GetSpacing(), 
                                                                                        reference_physical_size)]
    reference_origin = image.GetOrigin()
    reference_direction = image.GetDirection()

    reference_size = [round(sz * resize_factor) for sz, resize_factor in zip(image.GetSize(), resize_factor_list)] 
    reference_spacing = [phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

    transform = sitk.AffineTransform(dimension)
#     transform.SetMatrix(image.GetDirection())
    transform.SetMatrix((1, 0, 0, 0, 1, 0, 0, 0, 1))
    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform([centered_transform, centering_transform])
    min_value = float(np.min(sitk.GetArrayFromImage(image)))
    
    if nearest:
        new_img = sitk.Resample(image, reference_image, centered_transform, sitk.sitkNearestNeighbor, min_value)
    else:
        # source_image, refrence, transform, interpolation method, default value
        new_img = sitk.Resample(image, reference_image, centered_transform, sitk.sitkBSpline, min_value)
    
    return new_img

def write_series_to_path(target_image, original_sample_path, target_path, slice_thickness):
    tags_to_copy = ["0010|0010", # Patient Name
                    "0010|0020", # Patient ID
                    "0010|0030", # Patient Birth Date
                    "0020|000D", # Study Instance UID, for machine consumption
                    "0020|0010", # Study ID, for human consumption
                    "0008|0020", # Study Date
                    "0008|0030", # Study Time
                    "0008|0050", # Accession Number
                    "0008|0060"  # Modality
    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    direction = target_image.GetDirection()
    
    try:
        series_tag_value = reader.GetMetaData(0,"0008|103e")
    except RuntimeError:
        series_tag_value = "tag_None"
    
    original_image = sitk.ReadImage(original_sample_path)
    original_key_tuple = original_image.GetMetaDataKeys()
    original_tag_values = [(tag, original_image.GetMetaData(tag)) for tag in original_key_tuple]
    series_tag_values = [(k, original_image.GetMetaData(k)) for k in tags_to_copy if original_image.HasMetaDataKey(k)] + \
                     [("0008|0031",modification_time), # Series Time
                      ("0008|0021",modification_date), # Series Date
                      #("0008|0008","DERIVED\\SECONDARY"), # Image Type
                      #("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                      ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                      ("0008|103e", series_tag_value + " Processed-SimpleITK")]

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    
    os.makedirs(target_path, exist_ok=True)
    target_image_depth = target_image.GetDepth()
    series_instance_uid = os.path.basename(target_path)

    for index in range(target_image_depth):
        image_slice = target_image[:, :, index]
        # Tags shared by the series.
        
        for tag, value in original_tag_values:
            try:
                image_slice.SetMetaData(tag, value)
            except:
                continue
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Setting the type to CT preserves the slice location.
        image_slice.SetMetaData("0008|0060", "CT")  # set the type to CT so the thickness is carried over
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str, target_image.TransformIndexToPhysicalPoint((0,0,index))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(target_image_depth - index)) # Instance Number
        image_slice.SetMetaData("0018|0050", str(slice_thickness)) # set series slice thickness
        image_slice.SetMetaData("0020|000E", series_instance_uid)
        image_slice.SetMetaData("0020|000D", series_instance_uid)
        
        writer.SetFileName(f'{target_path}/{target_image_depth - index:04}.dcm')
        writer.Execute(image_slice)
        
def get_parent_dir_name(path, level=1):

    path_spliter = os.path.sep
    abs_path = os.path.abspath(path)

    return abs_path.split(path_spliter)[-(1 + level)]