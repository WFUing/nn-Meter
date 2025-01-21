import os

from nn_meter.builder.backends import BaseProfiler
import tensorflow as tf
import time
import docker


class TFProfiler(BaseProfiler):
    use_gpu = False

    def __init__(self, dst_kernel_path, benchmark_model_path, graph_path='', dst_graph_path='', serial='', num_threads=1, num_runs=50, warm_ups=10):
        """
        @params:
        graph_path: graph file. path on host server
        dst_graph_path: graph file. path on docker device
        kernel_path: dest kernel output file. path on docker device
        benchmark_model_path: path to benchmark_model on docker device
        """
        self._serial = serial
        self._graph_path = graph_path
        self._dst_graph_path = dst_graph_path
        self._dst_kernel_path = dst_kernel_path
        self._benchmark_model_path = benchmark_model_path
        self._num_threads = num_threads
        self._num_runs = num_runs
        self._warm_ups = warm_ups

    def profile(self, graph_path, preserve = False, clean = True, taskset = '70', close_xnnpack = False, **kwargs):
        """
        @params:
        preserve: tf file exists in remote dir. No need to push it again.
        clean: remove tf file after running.
        """
        model_name = os.path.basename(graph_path)
        remote_graph_path = os.path.join(self._dst_graph_path, model_name)
        client = docker.from_env()
        container = client.containers.run(
            "3e9c60a9a2f5",
            command=["--model_path", remote_graph_path, "--num_threads", self._num_threads , "--num_runs", self._num_runs,
                     "--warmup_runs", self._warm_ups],
            volumes={
                self._graph_path: {"bind": self._dst_graph_path, "mode": "rw"}},
            detach=True  # 后台运行
        )

        full_logs = container.logs()
        return full_logs
