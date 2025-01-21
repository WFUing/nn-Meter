import os
from nn_meter.builder import builder_config
from nn_meter.builder.kernel_predictor_builder import build_predictor_by_data

workspace = "/home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/debug"
builder_config.init(workspace)

kernel_type = "conv-bn-relu"
mark = "test"
backend = "my_backend"
error_threshold = 0.1

# 提取训练特征和目标值
cfgs_path = os.path.join(workspace, "predictor_build", "results", f"{kernel_type}_{mark}.json")
lats_path = os.path.join(workspace, "predictor_build", "results", f"profiled_{kernel_type}.json")
kernel_data = (cfgs_path, lats_path)

# 构建延迟预测器
predictor, acc10, error_configs = build_predictor_by_data(
    kernel_type, kernel_data, backend, error_threshold=error_threshold, mark=mark,
    save_path=os.path.join(workspace, "predictor_build", "results")
)
print(f"迭代 0: ±10% 准确率为 {acc10}，误差较大数据量：{len(error_configs)}")
