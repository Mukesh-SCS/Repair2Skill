# ==================== scripts/model_optimization.py ====================
import torch
import torch.nn as nn
import torch.quantization as quantization
from scripts.train_part_detector import FurnitureRepairModel
import time
import numpy as np


class ModelOptimizer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = FurnitureRepairModel()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def quantize_model(self, output_path):
        """Quantize model for faster inference"""

        # Dynamic quantization
        quantized_model = quantization.quantize_dynamic(
            self.model,
            {nn.Linear},
            dtype=torch.qint8
        )

        torch.save(quantized_model.state_dict(), output_path)
        print(f"Quantized model saved to: {output_path}")

        return quantized_model

    def prune_model(self, output_path, sparsity=0.3):
        """Prune model to reduce size"""
        import torch.nn.utils.prune as prune

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=sparsity)
                prune.remove(module, 'weight')

        torch.save(self.model.state_dict(), output_path)
        print(f"Pruned model saved to: {output_path}")

        return self.model

    def benchmark_models(self, input_size=(1, 3, 224, 224)):
        """Benchmark original, quantized, and pruned models"""

        dummy_input = torch.randn(input_size)

        models = {
            "Original": self.model,
            "Quantized": self.quantize_model("./models/damage_detection/quantized_model.pth"),
            "Pruned": self.prune_model("./models/damage_detection/pruned_model.pth")
        }

        results = {}

        for name, model in models.items():
            times = []
            for _ in range(20):
                start = time.time()
                with torch.no_grad():
                    model(dummy_input)
                end = time.time()
                times.append(end - start)

            avg_time = np.mean(times)
            results[name] = round(avg_time * 1000, 2)  # ms

        print("\n--- Inference Benchmark Results (ms) ---")
        for model_name, latency in results.items():
            print(f"{model_name}: {latency} ms")

        return results


if __name__ == "__main__":
    optimizer = ModelOptimizer("./models/damage_detection/part_detector.pth")
    optimizer.benchmark_models()
