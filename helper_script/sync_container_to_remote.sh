#! /bin/bash

# Check that the PROJECT_NAME is sourced
if [ "$PROJECT_NAME" != "recreating_ensemble_everything_everywhere" ]; then
    echo "helper_scripts/project_variables.sh is not sourced"
    exit 1
fi

# Sync the singularity container to the remote directory
REMOTE_DIR="${USERNAME}@${REMOTE_SERVER}:${SINGULARITY_STORAGE_DIR}"

# Sync the container to the remote directory
rsync -avz --progress \
    --delete \
    ./singularity/ $REMOTE_DIR
echo "Container synced to remote directory"

