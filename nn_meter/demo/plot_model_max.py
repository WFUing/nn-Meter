import matplotlib.pyplot as plt
import numpy as np

# 模型名称
models = [
    'AlexNets', 'VGGs', 'DenseNets', 'ResNets', 'SqueezNets',
    'GoogleNets', 'MobileNetv1s', 'MobileNetv2s', 'MobileNetv3s',
    'MnasNets', 'ProxylessNass', 'NASBench201'
]

# 移动CPU延迟(ms) - 最大值
cpu_latency_max = [
    494.4, 10289, 431.6, 1921.7, 524.9,
    274.6, 140.0, 211.0, 78.4,
    99.3, 195.9, 27.9
]

# 移动GPU延迟(ms) - 最大值
gpu_latency_max = [
    81.7, 1278, 69.5, 329.5, 72.2,
    49.0, 28.8, 37.0, 18.6,
    24.1, 72.2, 8.3
]

# Intel VPU延迟(ms) - 最大值
vpu_latency_max = [
    47.3, 1467, 70.7, 145.5, 57.3,
    24.4, 37.0, 86.1, 70.8,
    60.9, 77.8, 6.4
]

# 创建图表
plt.figure(figsize=(8, 6))

# 移动CPU
plt.scatter(models, cpu_latency_max, color='g', label='CPU Max (Pixel4 CortexA76, TFLite v2.1)')

# 移动GPU
plt.scatter(models, gpu_latency_max, color='y', label='GPU Max (Xiaomi Mi9 Adreno 640, TFLite v2.1)')

# Intel VPU
plt.scatter(models, vpu_latency_max, color='m', label='VPU Max (Intel NCS2 MyriadX, OpenVINO 2019R2)')

# 添加图例
plt.legend()

# 添加标题和标签
plt.title('Maximum Latency on Edge Devices')
plt.xlabel('Models')
plt.ylabel('Latency (ms)')

# 旋转x轴标签以便阅读
plt.xticks(rotation=90)

# 显示图表
plt.tight_layout()
# plt.show()
plt.savefig("model_latency_scatter.png")