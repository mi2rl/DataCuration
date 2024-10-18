# brainmeta_prediction.sh 파일의 위치는 "nnunetv2" 폴더와 level에 위치해야합니다.
# 모든 경로는 절대 경로로 표기 되어야 합니다.

CURRENT_DIR=$(dirname $(realpath $0))
###########################################################################################################
# manual typing
INPUT_DIR="$CURRENT_DIR/_inputs"
OUTPUT_DIR="$CURRENT_DIR/_results" 
GPUS="0"
###########################################################################################################

pip install -e $CURRENT_DIR/

mkdir -p "$CURRENT_DIR/nnUNet_raw"
mkdir -p "$CURRENT_DIR/nnUNet_preprocessed"
export nnUNet_raw="$CURRENT_DIR/nnUNet_raw"
export nnUNet_preprocessed="$CURRENT_DIR/nnUNet_preprocessed"
export nnUNet_results="$CURRENT_DIR/nnUNet_results"
export MKL_SERVICE_FORCE_INTEL=1

STAGE1_RESULTS="$OUTPUT_DIR/previous_stage_results"

if ls $OUTPUT_DIR/*.nii.gz 1> /dev/null 2>&1; then
    echo "Error: .nii.gz files already exist in the results folder. empty the results folder and try again."
    exit 1
fi

for file in $INPUT_DIR/*.nii.gz; do
    if [[ ! $file =~ "_0000.nii.gz" ]]; then
        mv "$file" "${file%.nii.gz}_0000.nii.gz"
    fi
done

# stage 1
CUDA_VISIBLE_DEVICES=$GPUS \
nnUNetv2_predict \
-i $INPUT_DIR \
-o $STAGE1_RESULTS \
-d 1 \
-f 0 1 2 3 4 \
-c 3d_lowres \
--save_probabilities

# stage 2
CUDA_VISIBLE_DEVICES=$GPUS \
nnUNetv2_predict \
-i $INPUT_DIR \
-o $OUTPUT_DIR \
-prev_stage_predictions $STAGE1_RESULTS \
-d 1 \
-f 0 1 2 3 4 \
-c 3d_cascade_fullres




