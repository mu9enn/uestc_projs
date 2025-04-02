import torch

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())
print("GPU Count:", torch.cuda.device_count())
print("Current GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# import paddle
# print(paddle.__version__)  # 应输出 2.5.2
# print(paddle.is_compiled_with_cuda())  # 应返回 True
# import google.protobuf
# print(google.protobuf.__version__)  # 应 >= 3.20.2