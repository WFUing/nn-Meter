# 使用官方 TensorFlow 镜像作为基础镜像
FROM tensorflow/tensorflow:2.6.0

# 设置工作目录
WORKDIR /app

# 复制本地 Python 脚本和模型文件到容器中
COPY tf_model_benchmark.py /app/benchmark_model.py

# 设置环境变量（可以根据需要配置）

# 设置默认命令运行 Python 脚本
ENTRYPOINT ["python", "benchmark_model.py"]
