# Prometheus exporter untuk monitoring model
from prometheus_client import start_http_server, Gauge, Counter
import psutil
import time
import random

# =============================
# PROMETHEUS METRICS
# =============================

system_cpu_usage = Gauge(
    "system_cpu_usage_percent",
    "CPU usage percentage of the system"
)

system_memory_usage = Gauge(
    "system_memory_usage_percent",
    "Memory usage percentage of the system"
)

system_disk_usage = Gauge(
    "system_disk_usage_percent",
    "Disk usage percentage on root partition"
)

system_process_count = Gauge(
    "system_process_count",
    "Number of running processes"
)

# REQUEST TOTAL (Counter)
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests received"
)

# LICENSE (Gauge)
license_remaining = Gauge(
    "license_remaining",
    "Remaining application licenses"
)

# =============================
# INITIALIZE PSUTIL
# =============================
psutil.cpu_percent(interval=None)

def collect_metrics():
    # CPU
    cpu = psutil.cpu_percent(interval=None)
    system_cpu_usage.set(cpu)

    # Memory
    memory = psutil.virtual_memory()
    system_memory_usage.set(memory.percent)

    # Disk
    disk = psutil.disk_usage("/")
    system_disk_usage.set(disk.percent)

    # Process count
    system_process_count.set(len(psutil.pids()))

    # Dummy request (simulasi trafik)
    http_requests_total.inc(random.randint(1, 5))

    # Dummy license (simulasi berkurang)
    current_license = max(0, 10 - int(time.time() / 60) % 10)
    license_remaining.set(current_license)

    print(
        f"CPU={cpu}% | MEM={memory.percent}% | "
        f"REQ={http_requests_total._value.get()} | "
        f"LIC={current_license}"
    )

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    start_http_server(9100)
    print("Prometheus exporter running at http://localhost:9100/metrics")

    while True:
        collect_metrics()
        time.sleep(5)
