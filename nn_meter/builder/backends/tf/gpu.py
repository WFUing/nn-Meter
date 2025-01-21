from nn_meter.builder.backends.interface import BaseProfiler
import tensorflow as tf
import time


class TFGPUProfiler(BaseProfiler):
    def __init__(self, model_path, input_shape, batch_size=1, num_runs=50, warmup_runs=10, **kwargs):
        self.model_path = model_path
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs

    def profile(self, model_path, **kwargs):
        # 设置 TensorFlow 优先使用 GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        # 加载模型
        model = tf.keras.models.load_model(model_path)
        input_shape = model.input_shape[1:]
        dummy_input = tf.random.uniform([1, *input_shape])

        # 预热
        for _ in range(self.warmup_runs):
            _ = model(dummy_input)

        # 测试时延
        start_time = time.time()
        for _ in range(self.num_runs):
            _ = model(dummy_input)
        end_time = time.time()

        avg_latency = (end_time - start_time) * 1000 / self.num_runs  # 毫秒
        return f"Average GPU latency: {avg_latency:.2f}ms"


class TFGPUProfiler(TFProfiler):
    use_gpu = True

