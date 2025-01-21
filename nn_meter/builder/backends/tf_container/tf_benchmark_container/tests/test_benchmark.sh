python tf_model_benchmark.py --model_path /home/wds/zhitai/graduate/projects/nn-Meter/nn_meter/builder/backends/tf_container/predictor_build/kernels/conv-bn-relu_test_PRG6AR.keras --num_threads 8 --num_runs 100 --warmup_runs 10

docker cp /home/wds/zhitai/graduate/projects/nn-Meter/data/pb_models/resnet18_0.pb 3e9c60a9a2f5:/app/models/resnet18_0.pb

docker run -d -v /home/wds/zhitai/graduate/projects/nn-Meter/data/pb_models:/app/models 3e9c60a9a2f5 --model_path /app/models/resnet18_0.pb --num_threads 8 --num_runs 100 --warmup_runs 10
