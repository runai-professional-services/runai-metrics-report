# #### Rows:
# workloads

# #### Columns:
# Project
# User  
# Job Name  
# GPU Hours  
# Memory Hours (GB)  
# CPU Hours  
# GPU Allocated  
# GPU Utilization % - Peak  
# GPU Utilization % - Average  
# CPU Utilization (Cores) - Peak  
# CPU Utilization (Cores) - Average  
# CPU Memory (GB) - Peak  
# CPU Memory (GB) - Average


import datetime

from runai.client import RunaiClient
from pydantic import BaseModel, Field
from typing import Optional


class Measurement(BaseModel):
    type: str
    labels: Optional[dict] = Field(default=None)
    values: list[dict]


class WorkloadMetric(BaseModel):
    type: str
    is_static_value: bool = Field(default=False)
    conversion_factor: float = Field(default=1.0)  # For GB conversion etc.

    def calculate(self, measurement: Measurement, start_time: str) -> tuple[float, float]:
        timestamp_prev = start_time
        total = 0.0
        peak = 0.0

        for item in measurement.values:
            timestamp = datetime.datetime.fromisoformat(item["timestamp"])
            prev_timestamp = datetime.datetime.fromisoformat(str(timestamp_prev))
            time_diff = (timestamp - prev_timestamp).total_seconds()

            value = float(item["value"])
            if not self.is_static_value:
                total += value * time_diff / 3600 * self.conversion_factor
            else:
                total += value * time_diff / 3600

            peak = max(peak, value)
            timestamp_prev = item["timestamp"]

        return total, peak


client = RunaiClient(
    client_id="api-test",
    client_secret="2I84hkD7HLxL1cPYD5nkBRl6p1EEmCUA",
    runai_base_url="https://envinaclick.run.ai"
)

# # start time 14 days ago in iso format
end = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = (datetime.datetime.fromisoformat(end) - datetime.timedelta(days=14)).isoformat()

metrics = ["GPU_ALLOCATION", "GPU_UTILIZATION", "GPU_MEMORY_USAGE_BYTES", "CPU_REQUEST_CORES", "CPU_USAGE_CORES", "CPU_MEMORY_USAGE_BYTES"]
# metrics = ["CPU_REQUEST_CORES", "CPU_USAGE_CORES", "CPU_MEMORY_USAGE_BYTES"]
# Define metrics configuration
metrics_config = {
    "GPU_ALLOCATION": WorkloadMetric(type="GPU_ALLOCATION", is_static_value=True),
    "CPU_REQUEST_CORES": WorkloadMetric(type="CPU_REQUEST_CORES", is_static_value=True),
    "CPU_USAGE_CORES": WorkloadMetric(type="CPU_USAGE_CORES"),
    "GPU_UTILIZATION": WorkloadMetric(type="GPU_UTILIZATION"),
    "CPU_MEMORY_USAGE_BYTES": WorkloadMetric(type="CPU_MEMORY_USAGE_BYTES", conversion_factor=1/(1024**3)),  # Convert to GB
    "GPU_MEMORY_USAGE_BYTES": WorkloadMetric(type="GPU_MEMORY_USAGE_BYTES", conversion_factor=1/(1024**3)),  # Convert to GB
}
workloads = client.workloads.all(filter_by="name==my-training")["workloads"]
print(workloads)
for workload in workloads:
    print("########################################################################")
    print(f"Workload ID: {workload['id']}")
    print(f"Project: {workload['projectName']}")
    print(f"Project ID: {workload['projectId']}")
    print(f"User: {workload['submittedBy']}")
    print("########################################################################")
    metrics = client.workloads.get_workload_metrics(
        workload_id=workload["id"],
        start=start,
        end=end,
        metric_type=metrics,
        number_of_samples=1000,
    )
    print(metrics)
    print("########################################################################")
    measurements = metrics["measurements"]
    cpu_hours = 0
    cpu_allocated = 0
    gpu_hours = 0
    gpu_allocated = 0
    cpu_memory_gb = 0
    gpu_memory_gb = 0
    cpu_utilization_peak = 0
    cpu_utilization_average = 0
    gpu_utilization_peak = 0
    gpu_utilization_average = 0
# Replace the existing calculation loop with:
    total_time_hours = (datetime.datetime.fromisoformat(end) - datetime.datetime.fromisoformat(start)).total_seconds() / 3600

    for measurement in measurements:
        m = Measurement(**measurement)
        if m.type in metrics_config:
            metric = metrics_config[m.type]
            total, peak = metric.calculate(m, start)
            
            if m.type == "CPU_USAGE_CORES":
                cpu_hours = total
                cpu_utilization_peak = peak
                cpu_utilization_average = total / total_time_hours * 3600
            elif m.type == "GPU_UTILIZATION":
                gpu_hours = total
                gpu_utilization_peak = peak
                gpu_utilization_average = total / total_time_hours * 3600
            elif m.type == "GPU_ALLOCATION":
                gpu_allocated = total
            elif m.type == "CPU_REQUEST_CORES":
                cpu_allocated = total
            elif m.type == "CPU_MEMORY_USAGE_BYTES":
                cpu_memory_gb = total
            elif m.type == "GPU_MEMORY_USAGE_BYTES":
                gpu_memory_gb = total
    print(f"CPU Hours: {cpu_hours}")
    print(f"CPU Allocated Hours: {cpu_allocated}")
    print(f"GPU Hours: {gpu_hours}")
    print(f"GPU Allocated: {gpu_allocated}")
    print(f"CPU Memory (GB): {cpu_memory_gb}")
    print(f"GPU Memory (GB): {gpu_memory_gb}")
    print(f"CPU Utilization (Cores) - Peak: {cpu_utilization_peak}")
    print(f"CPU Utilization (Cores) - Average: {cpu_utilization_average}")
    print(f"GPU Utilization % - Peak: {gpu_utilization_peak}")
    print(f"GPU Utilization % - Average: {gpu_utilization_average}")
print("########################################################################")