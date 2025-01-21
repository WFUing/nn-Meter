import logging
import os

from nn_meter.builder.backends import BaseBackend
from nn_meter.builder.backends.tf.tf_parser import TFParser
from nn_meter.builder.backends.tf.tf_profiler import TFProfiler

logging = logging.getLogger("nn-Meter")


class TensorFlowBackend(BaseBackend):
    profiler_class = TFProfiler
    parser_class = TFParser

    def update_configs(self):
        super().update_configs()

    def convert_model(self, model_path, save_path, input_shape=None):
        import os
        import shutil
        from onnx_tf.backend import prepare
        import onnx

        # 检查路径
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        dest_model_path = os.path.join(save_path, model_name + ".h5")

        if model_path.endswith(".onnx"):
            onnx_model = onnx.load(model_path)
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(dest_model_path)
        elif model_path.endswith(".h5") or model_path.endswith(".savedmodel"):
            shutil.copyfile(model_path, dest_model_path)
        else:
            raise ValueError("Unsupported model format")

        return dest_model_path

    def profile(self, converted_model, metrics=['latency'], **kwargs):
        # 获取输入形状和其他参数
        input_shape = kwargs.get('input_shape')
        batch_size = kwargs.get('batch_size', 1)
        num_runs = kwargs.get('num_runs', 50)
        warmup_runs = kwargs.get('warmup_runs', 10)

        profiler = self.profiler_class(
            model_path=converted_model,
            input_shape=input_shape,
            batch_size=batch_size,
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        output = profiler.profile()
        parser = self.parser_class()
        results = parser.parse(output).results.get(metrics)
        return results

    def test_connection(self):
        logging.info("TensorFlow backend is ready.")

