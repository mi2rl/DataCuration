import subprocess
import os
current_file_path = os.path.dirname(os.path.abspath(__file__))

import argparse
# argparse를 사용하여 명령줄 인자를 받기
parser = argparse.ArgumentParser(description="Process DICOM to NIfTI conversion and run nnU-Net")
parser.add_argument('--input_dcm', required=True, help='Path to the DICOM input folder')
parser.add_argument('--gpu_num', required=True, help='Path to the DICOM input folder')

args = parser.parse_args()

# 인자를 변수로 할당
input_dcm = args.input_dcm
gpu_num = args.gpu_num

os.system(f'pip install -e {current_file_path}')


import numpy as np
from glob import glob
import dicom2nifti
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from utils import calculate_intensity_distribution
import os
import matplotlib.pyplot as plt
import pydicom
import statistics
import csv
from utils import read_nifti_file,cos_similarity,determine_plane,calculate_intensity_in_direction,load_nii_with_simpleitk,display_slice_with_mask
from utils import list_files_by_creation_time

# List all files in the '../dcm_input' directory
modified_paths = os.listdir(input_dcm)

# Add '../' prefix to each item in the list
modified_paths = [os.path.join(input_dcm, path) for path in modified_paths]

modified_paths





curation_out_path = '../inputNii'

for in_path in tqdm(modified_paths):
    path_parts = in_path.split('/')


    
    out_name = path_parts[-1]
    print(out_name)
    # 슬래시(/)를 사용하여 경로를 다시 합칩니다
    out_path = '/'.join(path_parts)
    
    input_directory = in_path
    
    dicom2nifti.convert_directory(input_directory, curation_out_path)
    nii_time_list = list_files_by_creation_time(curation_out_path)
    
    rename = nii_time_list[-1].split('/')
    rename.pop()
    out_path = '/'.join(rename)
    
    rename = os.path.join(out_path,out_name+'.nii.gz')
  
    
    original_name = nii_time_list[-1]
    os.rename(original_name,rename)



subprocess.run(['python', 'run_prediction.py', '--input_dir', '../inputNii', '--output_dir', '../outputNii','--gpu_num',gpu_num])





# 파일 경로 설정
enhanced_ct_path = '../inputNii'
enhanced_mask_path = '../outputNii'

# nonenhance histo
non_enhanve_avg_intensity = []
enhance_avg_intensity_100down_path = []
enhance_avg_intensity_100up= []
avg_intensity = []
non_enhanve_avg_intensity = []
enhanve_avg_intensity = []


# nonenhance 평균 강도 
for file_name in tqdm(os.listdir(enhanced_ct_path), desc="Processing files"):
    try:
        # .nii.gz 파일만 처리
        if not file_name.endswith('.nii.gz'):
            continue
            
        maskname = file_name.replace('_0000', '')
        maskfile = os.path.join(enhanced_mask_path, maskname)
        
        
        enhanced_ct = read_nifti_file(os.path.join(enhanced_ct_path, file_name))
        enhanced_mask = read_nifti_file(maskfile)
        print(os.path.join(enhanced_ct_path, file_name))
        print(maskfile)
        
        
        if enhanced_ct is not None and enhanced_mask is not None:
            mask_array = sitk.GetArrayFromImage(enhanced_mask)
            img_array = sitk.GetArrayFromImage(enhanced_ct)
            # 마스크 데이터에서 유효한 레이블 확인
            unique_labels = np.unique(mask_array)

            
            sequence_intensities = []
            for slice_index in range(mask_array.shape[0]):
                
                label_mask = mask_array[slice_index] == 4 # 레이블이 1인 경우가 liver 영역입니다.
                if np.any(label_mask):
                    labeled_intensities = img_array[slice_index][label_mask]
                    average_intensity = np.mean(labeled_intensities)
                    sequence_intensities.append(average_intensity)
            
            if sequence_intensities:
                avg_sequence_intensity = int(round(statistics.mean(sequence_intensities)))
                avg_intensity.append(avg_sequence_intensity)
                print(avg_sequence_intensity)
                if avg_sequence_intensity < 100:
                    non_enhanve_avg_intensity.append(avg_sequence_intensity)

                    enhance_avg_intensity_100down_path.append(maskname)
                else:
                    enhance_avg_intensity_100up.append(maskname)
                    enhanve_avg_intensity.append(avg_sequence_intensity)
                    
    except Exception as e:
        print(f"Error processing file {ct_file_path}: {e}")
        continue

# 결과 출력
print("CT files with average intensity below 100:")
for path in enhance_avg_intensity_100down_path:
    print(path)

print("CT files with average intensity upper 100:")
for path in enhance_avg_intensity_100up:
    print(path)



# 문자열 변환 함수
def transform_path(input_string):
    return input_string.replace('_', '/')

data = []  # Initialize a list to store the data

for folder_name in enhance_avg_intensity_100up:
    # 폴더 경로 변환
    folder_path = transform_path(folder_name).replace('.nii.gz', '')
    rename_folder_path = ''.join(folder_path.split('/'))
    
    folder_path = os.path.join(input_dcm,folder_path)
    plane_printed = False  # Initialize the flag variable
    dcm_files = os.listdir(folder_path)
    if not dcm_files:
        print(f"No DICOM files found in {folder_path}")
        continue
    
    mid_index = len(dcm_files) // 2
    dicom_file_path = os.path.join(folder_path, dcm_files[mid_index])
    dicom_file = pydicom.dcmread(dicom_file_path)
    # Get the Slice Thickness
    slice_thickness = dicom_file.SliceThickness if "SliceThickness" in dicom_file else 'N/A'

    # Get the Kernel
    kernel = dicom_file.ConvolutionKernel if "ConvolutionKernel" in dicom_file else 'N/A'
    sex = dicom_file.PatientSex if "PatientSex" in dicom_file else 'N/A'

    img = dicom_file_path.split('/')[-2]
    # Get the Image Orientation (Patient) tag
    if "ImageOrientationPatient" in dicom_file:
        if not plane_printed:
            image_orientation = dicom_file.ImageOrientationPatient
            plane = determine_plane(image_orientation)
            data.append({
                'img': img,
                'sex': sex,
                'Slice Thickness': slice_thickness,
                'Kernel': kernel,
                'Plane': plane,
                'enhance': 'enhance'
            })  
        plane_printed = True
    else:
        print("Image Orientation (Patient) tag is not found in the DICOM file.")

# 결과 출력 또는 저장 (필요에 따라 추가)
print(data)




for folder_name in enhance_avg_intensity_100down_path:
    # 폴더 경로 변환
    folder_path = transform_path(folder_name).replace('.nii.gz', '')
    rename_folder_path = ''.join(folder_path.split('/'))
    
    folder_path = os.path.join(input_dcm,folder_path)
    plane_printed = False  # Initialize the flag variable
    dcm_files = os.listdir(folder_path)
    if not dcm_files:
        print(f"No DICOM files found in {folder_path}")
        continue
    
    mid_index = len(dcm_files) // 2
    dicom_file_path = os.path.join(folder_path, dcm_files[mid_index])
    dicom_file = pydicom.dcmread(dicom_file_path)

    # Get the Slice Thickness
    slice_thickness = dicom_file.SliceThickness if "SliceThickness" in dicom_file else 'N/A'

    # Get the Kernel
    kernel = dicom_file.ConvolutionKernel if "ConvolutionKernel" in dicom_file else 'N/A'

    sex = dicom_file.PatientSex if "PatientSex" in dicom_file else 'N/A'


    img = dicom_file_path.split('/')[-2]

    # Get the Image Orientation (Patient) tag
    if "ImageOrientationPatient" in dicom_file:
        if not plane_printed:
            image_orientation = dicom_file.ImageOrientationPatient
            plane = determine_plane(image_orientation)
            data.append({
                'img': img,
                'sex' : sex,
                'Slice Thickness': slice_thickness,
                'Kernel': kernel,
                'Plane': plane,
                'enhance': 'non'
            })
            plane_printed = True
    else:
            print("Image Orientation (Patient) tag is not found in the DICOM file.")


# Write the data to a CSV file
csv_file_path = 'output.csv'
with open(csv_file_path, mode='w', newline='') as csv_file:
    fieldnames = ['img','sex','Slice Thickness','Kernel','Plane','enhance']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for entry in data:
        writer.writerow(entry)


