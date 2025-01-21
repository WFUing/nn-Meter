from nn_meter.builder import builder_config
from nn_meter.builder.nn_meter_builder import sample_and_profile_kernel_data

workspace = "/home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/debug"
builder_config.init(workspace)

kernel_type = "conv-bn-relu"
sample_num = 10
backend = "my_backend"
mark = "test"

kernel_data = sample_and_profile_kernel_data(kernel_type, sample_num=sample_num,
                                             backend=backend, sampling_mode='prior', mark=mark)

