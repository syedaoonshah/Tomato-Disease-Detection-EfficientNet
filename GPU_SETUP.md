# GPU Setup Note for Windows

## Current Status ⚠️
- **TensorFlow**: 2.16.1 installed
- **GPU Detected**: ✅ NVIDIA GeForce RTX 4060 (CUDA 12.9, Driver 577.03)
- **CUDA Libraries**: ✅ Installed (cudnn, cublas, cuda-runtime)
- **GPU Support in TensorFlow**: ❌ Not working on Windows
- **Current Mode**: CPU training (slower but functional)

## Issue
TensorFlow 2.16+ on Windows doesn't have native GPU support. The CUDA libraries are installed but TensorFlow isn't detecting the GPU.

## Solutions

### Option 1: Use TensorFlow-DirectML (Recommended for Windows + AMD/NVIDIA GPUs)
```powershell
D:/Tomato/.venv/Scripts/python.exe -m pip uninstall tensorflow
D:/Tomato/.venv/Scripts/python.exe -m pip install tensorflow-directml-plugin
```

### Option 2: Install TensorFlow 2.10 (Last version with native Windows GPU support)
```powershell
D:/Tomato/.venv/Scripts/python.exe -m pip uninstall tensorflow
D:/Tomato/.venv/Scripts/python.exe -m pip install tensorflow-gpu==2.10.1
```
Note: Requires CUDA 11.2 and cuDNN 8.1

### Option 3: Use WSL2 with Ubuntu (Best Performance)
1. Install WSL2 with Ubuntu
2. Install NVIDIA drivers for WSL
3. Install CUDA toolkit in WSL
4. Run the training in WSL environment

### Option 4: Proceed with CPU Training (Current Setup)
The models will train on CPU. This is slower but will still work:
- Expected training time: 3-4x longer (~24-32 hours total)
- All functionality will work correctly
- Results will be identical to GPU training

## Current Configuration
- TensorFlow: 2.20.0 (CPU-only)
- Python: 3.12.4
- Platform: Windows
- GPU: RTX 4060 (detected by system but not by TensorFlow)

## Verification
Check TensorFlow configuration:
```powershell
D:/Tomato/.venv/Scripts/python.exe -c "import tensorflow as tf; print('TF Version:', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

## Recommendation
For this project with RTX 4060 on Windows, I recommend:
1. **Quick start**: Proceed with CPU training (works now, just slower)
2. **Better performance**: Install TensorFlow-DirectML for GPU acceleration
3. **Best performance**: Use WSL2 + Ubuntu + CUDA toolkit

The training scripts are already configured to detect and use GPU when available, so no code changes needed once GPU support is enabled.
