#!/bin/bash

export USER_NAME="aplesner"
export PROJECT_NAME="Ensemble-everything-everywhere-recreating-results"
export PROJECT_STORAGE_DIR="/itet-stor/${USER_NAME}/net_scratch/projects_storage/${PROJECT_NAME}"
export SCRATCH_STORAGE_DIR="/scratch/${USER_NAME}/${PROJECT_NAME}"

export REMOTE_SERVER="tik42x.ethz.ch"
export CODE_STORAGE_DIR="/home/${USER_NAME}/code/${PROJECT_NAME}"
export DATA_STORAGE_DIR="${PROJECT_STORAGE_DIR}/data"
export MODEL_STORAGE_DIR="${PROJECT_STORAGE_DIR}/models"
export SINGULARITY_STORAGE_DIR="${PROJECT_STORAGE_DIR}/singularity"
