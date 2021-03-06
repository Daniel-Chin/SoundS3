import torch

HAS_CUDA = torch.cuda.is_available()
CUDA = torch.device("cuda:0")
CPU  = torch.device("cpu")
if HAS_CUDA:
    DEVICE = CUDA
    print('We have CUDA.')
    gpu_name = torch.cuda.get_device_name(
        torch.cuda.current_device(), 
    )
    with open('gpu_name.txt', 'w') as f:
        print(gpu_name, file=f)
else:
    DEVICE = CPU
    print("We DON'T have CUDA.")
