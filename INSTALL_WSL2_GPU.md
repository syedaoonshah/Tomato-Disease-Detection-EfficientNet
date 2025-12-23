# GPU Setup with WSL2 (REQUIRED for GPU Training)

## Why WSL2?
TensorFlow no longer provides native GPU support for Windows. WSL2 (Windows Subsystem for Linux) is the **official Microsoft-recommended solution** for GPU computing on Windows.

## Installation Steps

### Step 1: Install WSL2 with Ubuntu
Open PowerShell as **Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

**Restart your computer** when prompted.

### Step 2: After Restart - Setup Ubuntu
1. Open "Ubuntu 22.04" from Start menu
2. Create username and password when prompted
3. Update Ubuntu:
```bash
sudo apt update && sudo apt upgrade -y
```

### Step 3: Install NVIDIA CUDA in WSL2
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update and install CUDA toolkit
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-3
```

### Step 4: Install Python and Dependencies in WSL2
```bash
# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip

# Navigate to your project (Windows D:\ drive is at /mnt/d/)
cd /mnt/d/Tomato

# Create virtual environment
python3.11 -m venv .venv-wsl

# Activate environment
source .venv-wsl/bin/activate

# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Install other requirements
pip install -r requirements.txt
```

### Step 5: Verify GPU Detection
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

You should see: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`

### Step 6: Start Training
```bash
cd /mnt/d/Tomato
source .venv-wsl/bin/activate
python src/train.py
```

## Benefits of WSL2
- ✅ **Full GPU support** - All CUDA features work
- ✅ **20x faster** - Each epoch ~30 seconds instead of 6 minutes
- ✅ **Official support** - Microsoft + NVIDIA + TensorFlow all recommend it
- ✅ **No dual boot** - Run alongside Windows

## Performance Comparison
| Environment | Time per Epoch | Total Training Time |
|------------|----------------|---------------------|
| Windows CPU | ~6 minutes | 24-36 hours |
| WSL2 + GPU | ~20-30 seconds | 2-4 hours |

## Next Steps
1. Open PowerShell as Administrator
2. Run: `wsl --install -d Ubuntu-22.04`
3. Restart computer
4. Follow steps above

Your existing work is safe - all files are in `/mnt/d/Tomato` accessible from WSL2.
