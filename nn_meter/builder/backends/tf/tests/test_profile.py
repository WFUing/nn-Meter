import tensorflow as tf
import numpy as np
import datetime

# 加载或定义模型
model = tf.keras.applications.MobileNetV2(weights=None, input_shape=(224, 224, 3), classes=1000)

# 准备输入数据
input_data = np.random.rand(1, 224, 224, 3).astype('float32')

# 设置日志目录
log_dir = "logs/inference/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# 定义推理函数
@tf.function
def inference():
    return model(input_data)

# 启动 Profiler
tf.profiler.experimental.start(log_dir)

# 运行推理多次
for _ in range(100):
    inference()

# 停止 Profiler
tf.profiler.experimental.stop()
