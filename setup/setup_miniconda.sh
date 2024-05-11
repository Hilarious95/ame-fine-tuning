#!/bin/bash

# Ensure the script exits if any command fails
# set -e

# Script Initialization
echo "Starting setup..."

# Update and upgrade system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y
sudo apt-get install build-essential
sudo apt install nvidia-cuda-toolkit

# Download Miniconda for Python 3.10 explicitly
echo "Downloading Miniconda..."
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh

# Make the installer executable
chmod +x ~/miniconda.sh

# Install Miniconda
echo "Installing Miniconda..."
~/miniconda.sh -b -p $HOME/miniconda3 -u

# Create and activate new Conda environment
echo "Creating and activating new Conda environment 'ame_ft_env'..."
~/miniconda3/bin/conda create --name ame_ft_env python=3.10 -y

# Set up Conda init to integrate with shell
echo "Initializing Conda..."
~/miniconda3/bin/conda init
source ~/.bashrc
PATH="/opt/conda/bin:${PATH}"
conda activate ame_ft_env

# Install PyTorch, CUDA toolkit, and xformers
echo "Installing PyTorch, CUDA toolkit, and xformers..."
conda install pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers -y
pip3 install torch

# Check GPU architecture and install appropriate packages
echo "Checking GPU architecture..."
python3 -c 'import torch; major, minor = torch.cuda.get_device_capability(); print(f"Major GPU version: {major}, Minor GPU version: {minor}"); version_flag = "new" if major >= 8 else "old"; exec(open("install_unsloth.py").read())'

# Save the following Python script as 'install_unsloth.py' in the same directory as this script
# This Python script will handle conditional installation based on GPU version
echo '''
import os
os.system("pip3 install 'unsloth[colab-new]@git+https://github.com/unslothai/unsloth.git'")
if version_flag == "new":
    os.system("pip3 install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes")
else:
    os.system("pip3 install --no-deps xformers trl peft accelerate bitsandbytes")
''' > install_unsloth.py

# Install other Python packages without dependencies
echo "Installing additional Python packages..."
pip3 install --no-deps trl peft accelerate bitsandbytes

# Final setup verification
echo "Verifying installation..."
conda --version
python3 -c 'import torch; print(torch.__version__)'

# Output to indicate successful setup
echo "Setup completed successfully. Environment 'ame_ft_env' is ready for use."
