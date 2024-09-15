

# Reproducing results from Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness

The goal of this repository is to recreate the results from the paper "Ensemble everything everywhere: Multi-scale aggregation for adversarial robustness" (https://arxiv.org/pdf/2408.05446) in PyTorch. This work is done independently of the paper as a hobby project

For Figure 5 & 6 the authors have outputs from 54 layers, but I only found 53 which I have taken as being after the layers [input, conv1+bn1+relu+maxpool, 50 after each block from `resent_models.ResNet._make_layer`, avgpool] (cf. `resnet_models.ResNet_conv._forward_impl`).

## Code structure
I am coding on my personal device but running computations on a remote cluster. I use the scripts in `helper_script` to sync the code and organize the files (the storage server and code are not located on the same directory). If you prefer to just clone the repository then update the storage paths in `helper_script/project_variables.sh` for the model and data directory. They should nonetheless be updated to what you require.

## Using singularity
Build container: `sudo singularity build singularity/pytorch_container.sif singularity/pytorch_container.def`

Execute container with `train_classifier.py` on cluster: `singularity exec --nv --bind ${PROJECT_STORAGE_DIR} ${SINGULARITY_STORAGE_DIR}/pytorch_container.sif python train_classifier.py`

Remember to update helper_script/project_variables.sh with your own username etc.
