import tensorflow as tf
import numpy as np
import time
import os
import argparse
import keras
import json
tf.compat.v1.disable_eager_execution()

class TFModelBenchmark:
    def __init__(self, model_path, num_threads=4, num_runs=50, warmup_runs=10, use_gpu=True):
        self.model_path = model_path
        self.num_threads = num_threads
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.use_gpu = use_gpu
        self.session, self.input_tensor, self.output_tensor = self.create_session()

    def create_session(self):
        file_extension = os.path.splitext(self.model_path)[-1].lower()
        session_config = tf.compat.v1.ConfigProto()
        session_config.intra_op_parallelism_threads = self.num_threads
        session_config.inter_op_parallelism_threads = self.num_threads
        session_config.gpu_options.allow_growth = self.use_gpu

        if file_extension == ".pb":
            # 加载 .pb 模型
            graph = tf.Graph()
            with graph.as_default():
                graph_def = tf.compat.v1.GraphDef()
                with tf.io.gfile.GFile(self.model_path, "rb") as f:
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="")

                session = tf.compat.v1.Session(graph=graph, config=session_config)
                # 自动获取第一个操作的输出作为输入张量，最后一个操作的输出作为输出张量
                input_tensor = session.graph.get_operations()[0].outputs[0]
                output_tensor = session.graph.get_operations()[-1].outputs[0]

        else:
            # 加载 .keras 模型
            model = tf.keras.models.load_model(self.model_path)
            session = tf.compat.v1.Session(config=session_config)
            dir_name = os.path.dirname(self.model_path)
            model_name = os.path.basename(self.model_path)
            json_file_path = os.path.join(dir_name, 'input_info.json')
            with open(json_file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)  # 读取文件内容
                except json.JSONDecodeError:
                    data = {}  # 如果文件为空或无效，初始化为空字典
            # print(data.get(model_name)['input_tensor_shape'][0])
            input_tensor = tf.compat.v1.placeholder(tf.float32, shape=data.get(model_name)['input_tensor_shape'][0])
            output_tensor = tf.math.reduce_sum(input_tensor)

        return session, input_tensor, output_tensor

    def get_inputs_by_shapes(shapes, batch_size=1):
        if len(shapes) == 1:
            return keras.Input(shape=shapes[0], batch_size=batch_size)
        else:
            return [keras.Input(shape=shape, batch_size=batch_size) for shape in shapes]

    def run_benchmark(self):
        # 预热运行
        for _ in range(self.warmup_runs):
            self.session.run(self.output_tensor, feed_dict={self.input_tensor: self.generate_random_input()})

        # 正式测试
        total_time = 0
        for _ in range(self.num_runs):
            start_time = time.time()
            self.session.run(self.output_tensor, feed_dict={self.input_tensor: self.generate_random_input()})
            total_time += time.time() - start_time

        average_time = total_time / self.num_runs * 1000
        print(f"{average_time:.2f} ms")  # 输出以毫秒为单位的平均时间

    def generate_random_input(self):
        # 使用输入张量的 shape 生成随机输入数据
        input_shape = [dim if dim is not None else 1 for dim in self.input_tensor.shape.as_list()]
        return np.random.rand(*input_shape).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="TensorFlow Model Benchmark Tool")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the TensorFlow model (.pb file)")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for benchmarking")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of runs for benchmarking")
    parser.add_argument("--warmup_runs", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for benchmarking if available")
    args = parser.parse_args()

    benchmark = TFModelBenchmark(
        model_path=args.model_path,
        num_threads=args.num_threads,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        use_gpu=args.use_gpu
    )
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()

