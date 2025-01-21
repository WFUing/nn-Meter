import docker
import time

# 创建 Docker 客户端
client = docker.from_env()

# 启动 Docker 容器
container = client.containers.run(
    "d60029bb45b5",
    command=["--model_path", "/app/models/conv-bn-relu_test_1IC207", "--num_threads", "8", "--num_runs", "100", "--warmup_runs", "10"],
    volumes={"/home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/tf_container/predictor_build/kernels": {"bind": "/app/models", "mode": "rw"}},
    detach=True  # 后台运行
)

# 获取容器的日志输出
try:
    for line in container.logs(stream=True):
        print(line.decode('utf-8').strip())
except Exception as e:
    print(f"Error while getting logs: {str(e)}")

# 等待容器运行完成并停止
container.wait()

# 获取容器的最终状态
exit_code = container.wait()["StatusCode"]
print(f"Container exited with code {exit_code}")

# 获取完整的日志输出（如果容器已完成运行）
full_logs = container.logs()
print(full_logs.decode('utf-8'))
