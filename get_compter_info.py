import os
import platform
import subprocess
import torch

def get_gpu_info():
    gpu_info = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "Name": props.name,
                "Compute Capability": f"{props.major}.{props.minor}",
                "Total Memory (GB)": props.total_memory / 1e9,
                "Multiprocessors": props.multi_processor_count,
               # "Clock Rate (MHz)": props.clock_rate / 1e3,
               # "Memory Bus Width (bits)": props.memory_bus_width,
            })
    else:
        gpu_info.append({"Message": "No GPUs available."})
    return gpu_info

def get_cpu_info():
    return {
        "Processor": platform.processor(),
        "Cores": os.cpu_count(),
    }

def get_memory_info():
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem_total = int(next(line.split()[1] for line in lines if line.startswith("MemTotal"))) / 1e6
        mem_free = int(next(line.split()[1] for line in lines if line.startswith("MemFree"))) / 1e6
        return {
            "Total Memory (GB)": mem_total,
            "Free Memory (GB)": mem_free,
        }
    except FileNotFoundError:
        return {"Message": "Memory information is unavailable."}

def get_disk_info():
    try:
        disk_usage = subprocess.check_output(["df", "-h", "--output=source,size,used,avail"]).decode().splitlines()
        disk_info = [line.split() for line in disk_usage[1:]]
        return [{"Mountpoint": info[0], "Total Size": info[1], "Used": info[2], "Free": info[3]} for info in disk_info]
    except Exception:
        return [{"Message": "Disk information is unavailable."}]

def get_system_info():
    return {
        "OS": platform.system(),
        "OS Version": platform.version(),
        "OS Release": platform.release(),
        "Architecture": platform.architecture()[0],
        "Hostname": platform.node(),
    }

if __name__ == "__main__":
    print("=== GPU Information ===")
    for gpu in get_gpu_info():
        for key, value in gpu.items():
            print(f"{key}: {value}")
        print()
    
    print("=== CPU Information ===")
    for key, value in get_cpu_info().items():
        print(f"{key}: {value}")
    
    print("\n=== Memory Information ===")
    for key, value in get_memory_info().items():
        print(f"{key}: {value}")
    
    print("\n=== Disk Information ===")
    for disk in get_disk_info():
        for key, value in disk.items():
            print(f"{key}: {value}")
        print()
    
    print("=== System Information ===")
    for key, value in get_system_info().items():
        print(f"{key}: {value}")
