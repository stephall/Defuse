#!/bin/bash
################################################################################
### This script assumes that conda has been installed and aliased to 'conda'
################################################################################

# Check that at least one argument has beend passed
if ! [ $# -eq 1 ]
    then
        echo "Error: Please pass the name of the to be created (conda) environment as the only argument."
        exit 1
fi

# Use the first argument as the name
ENV_NAME=$1 

# Inform the user
echo "Creating a conda environment called '${ENV_NAME}'"
echo ""

# Determine the (absolute) base path to the folder containing conda (where anaconda or miniconda has been installed),
# use it to construct the absolute path to the conda.sh file, and source this file
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Create a conda environment
conda create -n $ENV_NAME python=3.9

# Activate the conda environment
conda activate $ENV_NAME

# Install pip using conda and upgrade it to the latest version
conda install pip
pip install --upgrade pip
    
# Install standard datascience modules
pip install numpy==1.26.2 matplotlib==3.8.2 pandas==2.1.3 scipy==1.11.4

# Install scikit-learn (to use SwissRoll and other datasets)
pip install scikit-learn==1.3.2

# Install pytorch
pip install torch==2.1.1

# Install ipykernel (for jupyter notebooks) and the ipywidgets (for additional functionality)
pip install ipykernel==6.20.2
pip install ipywidgets==8.0.2


############################################################################################################################################################
# Make the project itself a python package (to enable more convenient absolute imports)
# 1) Generate a setup.py file for the project with the ENV_NAME as name
# Remark: Enable echo to interpret backslash escapes (required for the two newline characters '\n\n' below)
echo -e "from setuptools import setup, find_packages\n\nsetup(name='${ENV_NAME}', version='1.0', packages=find_packages())" > setup.py

# 2) Install the project itself by its name and make it editable (=> use '-e' for editable)
# Remark: This requires the setup.py file created above
pip install -e .
############################################################################################################################################################

echo " "
echo "Installation done"

conda deactivate
