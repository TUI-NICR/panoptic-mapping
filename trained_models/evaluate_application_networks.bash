#!/bin/bash
set -x
set -e

# define paths ---------------------------------------------------------------------------------------------------------
BASEPATH_WEIGHTS=$(dirname $(realpath -s "$0"))
BASEPATH_EMSANET="${BASEPATH_WEIGHTS}/../emsanet"
# BASEPATH_EMSANET="/home/dase6070/gitlab/emsanet"

cd ${BASEPATH_EMSANET}

if [[ $(hostname) == *"makalu"* ]]; then
    # datasets
    BASEPATH_NYUV2="/mnt/training_gpu/dase6070/datasets/nicr-scene-analysis-datasets-nyuv2-v030"
    BASEPATH_HYPERSIM="/mnt/training_gpu/dase6070/datasets/nicr-scene-analysis-datasets-hypersim-v052"
    BASEPATH_SUNRGBD="/mnt/training_gpu/dase6070/datasets/nicr-scene-analysis-datasets-sunrgbd-v060"
    BASEPATH_SCANNET="/mnt/training_gpu/dase6070/datasets/nicr-scene-analysis-datasets-scannet-v051"

    BATCH_SIZE=16
elif [[ $(hostname) == *"apfel1"* ]]; then
    # datasets
    BASEPATH_NYUV2="/datasets_nas/nicr_scene_analysis_datasets/version_060/nyuv2"
    BASEPATH_HYPERSIM="/datasets_nas/nicr_scene_analysis_datasets/version_060/hypersim"
    BASEPATH_SUNRGBD="/datasets_nas/nicr_scene_analysis_datasets/version_060/sunrgbd"
    BASEPATH_SCANNET="/datasets_nas/nicr_scene_analysis_datasets/version_060/scannet"

    BATCH_SIZE=1
else
    # datasets
    BASEPATH_NYUV2="${BASEPATH}/../datasets/nyuv2"
    BASEPATH_HYPERSIM="${BASEPATH}/../datasets/hypersim"
    BASEPATH_SUNRGBD="${BASEPATH}/../datasets/sunrgbd"
    BASEPATH_SCANNET="${BASEPATH}/../datasets/scannet"

    BATCH_SIZE=1
fi


# define default args --------------------------------------------------------------------------------------------------
DEFAULT_ARGS="--no-pretrained-backbone --validation-only --validation-batch-size ${BATCH_SIZE} --wandb-mode disabled --results-basepath ./does_not_matter --semantic-class-weighting none --skip-sanity-check"


# application model ----------------------------------------------------------------------------------------------------
WEIGHTS_FP="${BASEPATH_WEIGHTS}/application/r34_NBt1D.pth"
ARGS_FP="${BASEPATH_WEIGHTS}/application/r34_NBt1D_argv.txt"
ARGS=$(cat ${ARGS_FP})

# validate on nyuv2 test split
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_NYUV2}:${BASEPATH_HYPERSIM}:${BASEPATH_SUNRGBD}:${BASEPATH_SCANNET}" --validation-split test:none:none:none
# validate on sunrgbd test split
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_NYUV2}:${BASEPATH_HYPERSIM}:${BASEPATH_SUNRGBD}:${BASEPATH_SCANNET}" --validation-split none:none:test:none
# validate on hypersim valid + test split
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_NYUV2}:${BASEPATH_HYPERSIM}:${BASEPATH_SUNRGBD}:${BASEPATH_SCANNET}" --validation-split none:valid:none:none
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_NYUV2}:${BASEPATH_HYPERSIM}:${BASEPATH_SUNRGBD}:${BASEPATH_SCANNET}" --validation-split none:test:none:none
# validate on scannet valid split (40 classes)
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_NYUV2}:${BASEPATH_HYPERSIM}:${BASEPATH_SUNRGBD}:${BASEPATH_SCANNET}" --validation-split none:none:none:valid --validation-scannet-subsample 100 --scannet-semantic-n-classes 40
# validate on scannet valid split (only 20 of 40 classes)
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_NYUV2}:${BASEPATH_HYPERSIM}:${BASEPATH_SUNRGBD}:${BASEPATH_SCANNET}" --validation-split none:none:none:valid --validation-scannet-subsample 100 --scannet-semantic-n-classes 40 --validation-scannet-benchmark-mode


# fine-tuned application models ---------------------------------------------------------------------------------------
# validate on nyuv2 test split
WEIGHTS_FP="${BASEPATH_WEIGHTS}/application_nyuv2_finetuned/r34_NBt1D.pth"
ARGS_FP="${BASEPATH_WEIGHTS}/application_nyuv2_finetuned/r34_NBt1D_argv.txt"
ARGS=$(cat ${ARGS_FP})
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_NYUV2}" --validation-split test

# validate on sunrgbd test split
WEIGHTS_FP="${BASEPATH_WEIGHTS}/application_sunrgbd_finetuned/r34_NBt1D.pth"
ARGS_FP="${BASEPATH_WEIGHTS}/application_sunrgbd_finetuned/r34_NBt1D_argv.txt"
ARGS=$(cat ${ARGS_FP})
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_SUNRGBD}" --validation-split test

# validate on hypersim valid + test split
WEIGHTS_FP="${BASEPATH_WEIGHTS}/application_hypersim_finetuned/r34_NBt1D.pth"
ARGS_FP="${BASEPATH_WEIGHTS}/application_hypersim_finetuned/r34_NBt1D_argv.txt"
ARGS=$(cat ${ARGS_FP})
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_HYPERSIM}" --validation-split valid
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_HYPERSIM}" --validation-split test

# validate on scannet valid split (40 classes)
WEIGHTS_FP="${BASEPATH_WEIGHTS}/application_scannet20_finetuned/r34_NBt1D.pth"
ARGS_FP="${BASEPATH_WEIGHTS}/application_scannet20_finetuned/r34_NBt1D_argv.txt"
ARGS=$(cat ${ARGS_FP})
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_SCANNET}" --validation-split valid

# validate on scannet valid split (only 20 of 40 classes)
WEIGHTS_FP="${BASEPATH_WEIGHTS}/application_scannet40_finetuned/r34_NBt1D.pth"
ARGS_FP="${BASEPATH_WEIGHTS}/application_scannet40_finetuned/r34_NBt1D_argv.txt"
ARGS=$(cat ${ARGS_FP})
python ${ARGS} ${DEFAULT_ARGS} --weights-filepath ${WEIGHTS_FP} --dataset-path "${BASEPATH_SCANNET}" --validation-split valid

cd -
