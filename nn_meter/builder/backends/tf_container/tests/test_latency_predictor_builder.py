from build.lib.nn_meter.builder import build_predictor_for_kernel
from nn_meter.builder import builder_config, build_latency_predictor
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases

# initialize builder config with workspace
builder_config.init(
    workspace_path="/home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/tf_container"
)

# from nn_meter.builder.backends import connect_backend
# backend = connect_backend(backend_name='tf_backend')

build_latency_predictor(backend='tf_backend')
