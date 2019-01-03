#!/bin/bash

# Initiate all cuda environment variables
if [ -d "/usr/local/cuda-9.0/bin/" ]; then
    echo "Initializing Cuda Environment Variables"
    export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
fi