#!/bin/bash
# Install CUDA 11.0 for TensorFlow 2.10 in WSL2

echo "Installing CUDA 11.0 for TensorFlow 2.10..."

# Remove any existing CUDA packages
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*" -y
sudo apt-get autoremove -y
sudo apt-get autoclean -y

# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.0.3/local_installers/cuda-repo-wsl-ubuntu-11-0-local_11.0.3-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-11-0-local_11.0.3-1_amd64.deb
sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-0-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda-11-0

# Install cuDNN 8
wget https://developer.download.nvidia.com/compute/cudnn/8.0.5/local_installers/cudnn-11.0-linux-x64-v8.0.5.39.tgz
tar -xzvf cudnn-11.0-linux-x64-v8.0.5.39.tgz
sudo cp cuda/include/cudnn*.h /usr/local/cuda-11.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.0/lib64
sudo chmod a+r /usr/local/cuda-11.0/include/cudnn*.h /usr/local/cuda-11.0/lib64/libcudnn*

# Set environment variables
echo 'export PATH=/usr/local/cuda-11.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "CUDA 11.0 installation complete!"
echo "Please run: source ~/.bashrc"
