import os
import numpy as np
from glob import glob
import argparse

def main(input_dir, output_dir, gpu_num):
    # Set the current working directory
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_file_path)
    print(f"Current working directory: {current_file_path}")

    # Set environment variables
    os.environ['PATH'] = f'{os.environ["PATH"]}:~/.local/bin'
    os.system(f'pip install -e {current_file_path}')
    os.environ['nnUNet_raw'] = f'{current_file_path}/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = f'{current_file_path}/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = f'{current_file_path}/nnUNet_results'

    os.makedirs(output_dir, exist_ok=True)

    # Rename files if necessary
    for i in glob(f'{input_dir}/*'):
        if not i.endswith('_0000.nii.gz'):
            os.rename(i, i.replace('.nii.gz', '_0000.nii.gz'))

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
    os.environ['MKL_THREADING_LAYER']='GNU'
    command = (
        f'CUDA_VISIBLE_DEVICES={gpu_num} '
        f'nnUNetv2_predict '
        f'-i {input_dir} '  # input_folder
        f'-o {output_dir} '  # output_folder
        f'-d 1 '  # dataset number
        f'-c 3d_lowres '  # configuration
        f'-npp 1 '
        f'-nps 1 '
    )

    os.system(command)

if __name__ == "__main__":
    # Argument parser to handle input arguments
    parser = argparse.ArgumentParser(description='Run nnUNetv2 prediction.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory for the files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the results.')
    parser.add_argument('--gpu_num', type=str, default='0', help='GPU number to use.')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.gpu_num)
