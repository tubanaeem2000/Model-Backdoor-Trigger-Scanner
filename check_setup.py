"""
check_setup.py - Step 1 test: are all libraries installed correctly?
Run:  python check_setup.py
"""
import sys

print("Python version:", sys.version.split()[0])

try:
    import torch, torchvision, streamlit, matplotlib, numpy
    print("PyTorch     :", torch.__version__)
    print("TorchVision :", torchvision.__version__)
    print("Streamlit   :", streamlit.__version__)
    print("Matplotlib  :", matplotlib.__version__)
    print("NumPy       :", numpy.__version__)
    print("GPU (CUDA) available:", torch.cuda.is_available(), "(CPU is fine for MNIST)")
    print("\nSetup OK - all libraries installed.")
except ImportError as e:
    print("\nMissing library:", e.name)
    print("Fix: pip install -r requirements.txt")
