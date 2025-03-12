import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Available GPUs:", torch.cuda.device_count())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
