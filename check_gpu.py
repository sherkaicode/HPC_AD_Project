import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  - Multiprocessors: {props.multi_processor_count}")
        print(f"  - CUDA Cores: {props.multi_processor_count * 64}")  # Approximation for NVIDIA GPUs
        print(f"  - Clock Rate: {props.clock_rate / 1e3:.2f} MHz")
        print(f"  - Memory Bus Width: {props.memory_bus_width} bits")
else:
    print("No GPUs available.")
