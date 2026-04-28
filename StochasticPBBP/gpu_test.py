
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
