from build.lib.nn_meter.builder import build_predictor_for_kernel
from nn_meter.builder import builder_config
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases

# initialize builder config with workspace
builder_config.init(
    workspace_path="/home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/tf_container"
)

# generate testcases
origin_testcases = generate_testcases()

from nn_meter.builder.backends import connect_backend
backend = connect_backend(backend_name='tf_backend')

# run testcases and collect profiling results
from nn_meter.builder import profile_models
profiled_results = profile_models(backend, origin_testcases, mode='ruletest')

print(profiled_results)

from nn_meter.builder.backend_meta.fusion_rule_tester import detect_fusion_rule

# determine fusion rules from profiling results
detected_results = detect_fusion_rule(profiled_results)

print(detected_results)

