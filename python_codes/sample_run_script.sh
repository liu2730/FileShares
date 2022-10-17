#!/bin/bash

export LD_LIBRARY_PATH=/home/liu_fl/Downloads/photon-master/cuda_codes/lib:/home/liu_fl/Downloads/photon-master/cuda_codes/lib64

simulation_type=$1

#python createNRRD.py

python create_sample_simulation_parameters.py

python batch_run_simulation.py ../sample-data/$simulation_type/parameters 1 1