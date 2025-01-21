import os

# 原始路径
path = '/home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/tf_container/predictor_build/kernels/conv-bn-relu_test_IRDUJL'

# 提取目录部分
directory_path = os.path.dirname(path)

# 提取最后的文件夹或文件名部分
file_name = os.path.basename(path)

print("目录路径:", directory_path)
print("文件/文件夹名:", file_name)
