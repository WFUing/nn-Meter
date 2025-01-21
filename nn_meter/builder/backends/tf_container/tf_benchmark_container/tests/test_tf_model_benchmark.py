from nn_meter.builder.backends.tf_container.tf_benchmark_container.tf_model_benchmark import TFModelBenchmark

benchmark = TFModelBenchmark(
    model_path="/home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/tf_container/predictor_build/kernels/conv-bn-relu_test_1IC207",
    num_threads=8,
    num_runs=100,
    warmup_runs=10,
    use_gpu=False
)

benchmark.run_benchmark()

