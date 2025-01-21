import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from nn_meter.builder.nn_modules import BaseBlock
from nn_meter.builder.nn_modules.torch_networks.blocks import (
    TorchBlock, ConvBnRelu, ConvBn, ConvRelu, ConvBlock, ConvHswish, DwConvBlock
)

class TorchModelBenchmark:
    def __init__(self, model_path, num_threads=4, num_runs=50, warmup_runs=10, use_gpu=True):
        self.model_path = model_path
        self.num_threads = num_threads
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.model = self.load_model()

    def load_model(self):
        file_extension = self.model_path.split('.')[-1].lower()

        if file_extension == "pt" or file_extension == "pth":
            # Load PyTorch .pt or .pth model
            model = torch.load(self.model_path, map_location=self.device)
            if isinstance(model, dict) and "model" in model:
                model = model["model"]
        elif file_extension == "torchscript":
            # Load TorchScript model
            model = torch.jit.load(self.model_path, map_location=self.device)
        else:
            raise ValueError(f"Unsupported model file format: {file_extension}")

        model.eval()
        model.to(self.device)
        return model

    def run_benchmark(self):
        # Generate random input based on the model's input shape
        input_tensor = self.generate_random_input()

        # Warm-up runs
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = self.model(input_tensor)

        # Benchmark runs
        total_time = 0
        for _ in range(self.num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.model(input_tensor)
            total_time += time.time() - start_time

        average_time = total_time / self.num_runs
        print(f"{average_time * 1000:.2f} ms")  # Output average time in milliseconds

    def generate_random_input(self):
        # Generate random input based on the model's expected input shape
        dummy_input_shape = (1, 3, 224, 224)  # Default to a common input shape
        if hasattr(self.model, "dummy_input_shape"):
            dummy_input_shape = self.model.dummy_input_shape
        return torch.rand(dummy_input_shape, device=self.device)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Model Benchmark Tool")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch model (.pt/.pth file)")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads for benchmarking")
    parser.add_argument("--num_runs", type=int, default=50, help="Number of runs for benchmarking")
    parser.add_argument("--warmup_runs", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for benchmarking if available")
    args = parser.parse_args()

    # Set the number of threads for PyTorch
    torch.set_num_threads(args.num_threads)

    benchmark = TorchModelBenchmark(
        model_path=args.model_path,
        num_threads=args.num_threads,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        use_gpu=args.use_gpu
    )
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
