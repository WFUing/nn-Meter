from nn_meter.builder.backends import BaseBackend
from nn_meter.builder.backend_meta.utils import Latency, ProfiledResults
import docker
import os
import re
import statistics


class TFContainerBackend(BaseBackend):

    def profile(self, graph_path, metrics=['latency'], num_measurements=2, num_threads=6, warm_ups=0, num_runs=1, **kwargs):

        model_path = os.path.dirname(graph_path)
        model_name = os.path.basename(graph_path)
        remote_graph_path = os.path.join("/app/models", model_name)
        client = docker.from_env()

        latencies = []

        for _ in range(num_measurements):
            container = client.containers.run(
                "53af1930159f",
                command=[
                    "--model_path", remote_graph_path,
                    "--num_threads", str(num_threads),
                    "--num_runs", str(num_runs),
                    "--warmup_runs", str(warm_ups),
                ],
                volumes={model_path: {"bind": "/app/models", "mode": "rw"}},
                detach=True  # Run in the background
            )

            # 等待容器运行完成并停止
            container.wait()
            # 获取容器的最终状态
            exit_code = container.wait()["StatusCode"]
            print(f"Container exited with code {exit_code}")
            full_logs = container.logs()
            print(full_logs)
            if isinstance(full_logs, bytes):
                full_logs = full_logs.decode('utf-8')

            match = re.search(r"(\d+\.\d+)\s*ms", full_logs)
            print(match)
            if match:
                latencies.append(float(match.group(1)))

        if latencies:
            avg = statistics.mean(latencies)  # Calculate average latency
            std = statistics.stdev(latencies)  # Calculate standard deviation
        else:
            avg, std = 0, 0  # Handle case where no latencies were recorded

        latency = Latency(avg, std)
        print(avg, std)
        return ProfiledResults({'latency': latency}).get(metrics)

