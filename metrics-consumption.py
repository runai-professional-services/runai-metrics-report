import datetime
import csv
import os
from runai.configuration import Configuration
from runai.api_client import ApiClient
from runai.runai_client import RunaiClient
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

    def calculate(self, measurement: Measurement) -> tuple[float, float]:
        total = 0.0
        peak = 0.0
        timestamp_prev = measurement.values[0]["timestamp"]

        for granular in measurement.values:
            timestamp = datetime.datetime.fromisoformat(granular["timestamp"])
            prev_timestamp = datetime.datetime.fromisoformat(str(timestamp_prev))
            time_diff = (timestamp - prev_timestamp).total_seconds()

            value = float(granular["value"])
            if not self.is_static_value:
                total += value * time_diff / 3600 * self.conversion_factor
            else:
                total += value * time_diff / 3600

            peak = max(peak, value)
            timestamp_prev = granular["timestamp"]

        return total, peak


# Initialize Run:AI client with new syntax
config = Configuration(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    runai_base_url=os.getenv('BASE_URL')
)

client = RunaiClient(ApiClient(config))

# Define the time range (last 24 hours)
end = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = (datetime.datetime.fromisoformat(end) - datetime.timedelta(days=1)).isoformat()

# Define metric types as a list of strings
metrics_types = [
    "GPU_ALLOCATION",
    "GPU_UTILIZATION",
    "GPU_MEMORY_USAGE_BYTES",
    "CPU_REQUEST_CORES",
    "CPU_USAGE_CORES",
    "CPU_MEMORY_USAGE_BYTES"
]

# Define metrics configuration
metrics_config = {
    "GPU_ALLOCATION": WorkloadMetric(type="GPU_ALLOCATION", is_static_value=True),
    "CPU_REQUEST_CORES": WorkloadMetric(type="CPU_REQUEST_CORES", is_static_value=True),
    "CPU_USAGE_CORES": WorkloadMetric(type="CPU_USAGE_CORES"),
    "GPU_UTILIZATION": WorkloadMetric(type="GPU_UTILIZATION"),
    "CPU_MEMORY_USAGE_BYTES": WorkloadMetric(type="CPU_MEMORY_USAGE_BYTES", conversion_factor=1/(1024**3)),  # Convert to GB
    "GPU_MEMORY_USAGE_BYTES": WorkloadMetric(type="GPU_MEMORY_USAGE_BYTES", conversion_factor=1/(1024**3)),  # Convert to GB
}

# Create CSV file with current timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"workload_metrics_{timestamp}.csv"

# Define CSV headers
csv_headers = [
    "Workload Name",
    "Workload ID",
    "Project",
    "Project ID",
    "User",
    "CPU Hours",
    "CPU Allocated Hours",
    "GPU Hours",
    "GPU Allocated",
    "CPU Memory (GB)",
    "GPU Memory (GB)",
    "CPU Utilization Peak (Cores)",
    "CPU Utilization Average (Cores)",
    "GPU Utilization Peak (%)",
    "GPU Utilization Average (%)",
    "Start Time",
    "End Time"
]

# Open CSV file for writing
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

    # Fetch all workloads using the new client syntax
    try:
        response = client.workloads.workloads.get_workloads()
        workloads_data = response.data.get('workloads', [])
        print(f"Total Workloads Found: {len(workloads_data)}")
    except Exception as e:
        print(f"Error fetching workloads: {e}")
        exit(1)

    # Iterate through workloads
    for workload in workloads_data:
        workload_id = workload.get('id')
        if not workload_id:
            continue

        # Fetch workload metrics
        try:
            metrics_response = client.workloads.workloads.get_workload_metrics(
                workload_id=workload_id,
                start=start,
                end=end,
                metric_type=metrics_types,
                number_of_samples=1000,
            )

            if not hasattr(metrics_response, 'data') or not metrics_response.data:
                print(f"No metrics data available for workload {workload.get('name', workload_id)}")
                continue

            metrics = metrics_response.data
        except Exception as e:
            print(f"Error fetching metrics for {workload.get('name', workload_id)}: {e}")
            continue

        if not metrics.get("measurements"):
            print(f"No measurements available for workload {workload.get('name', workload_id)}")
            continue

        # Initialize metrics
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

        # Process each measurement
        for measurement in metrics["measurements"]:
            if not measurement.get("values"):
                continue

            measurement_start_time = measurement["values"][0]["timestamp"]
            measurement_end_time = measurement["values"][-1]["timestamp"]
            total_time_hours = (datetime.datetime.fromisoformat(measurement_end_time) - datetime.datetime.fromisoformat(measurement_start_time)).total_seconds() / 3600

            m = Measurement(**measurement)
            if m.type in metrics_config:
                metric = metrics_config[m.type]
                total, peak = metric.calculate(m)

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

        # Write row to CSV
        writer.writerow({
            "Workload Name": workload.get('name', 'Unknown'),
            "Workload ID": workload_id,
            "Project": workload.get('projectName', 'Unknown'),
            "Project ID": workload.get('projectId', 'Unknown'),
            "User": workload.get('submittedBy', 'Unknown'),
            "CPU Hours": round(cpu_hours, 2),
            "CPU Allocated Hours": round(cpu_allocated, 2),
            "GPU Hours": round(gpu_hours, 2),
            "GPU Allocated": round(gpu_allocated, 2),
            "CPU Memory (GB)": round(cpu_memory_gb, 2),
            "GPU Memory (GB)": round(gpu_memory_gb, 2),
            "CPU Utilization Peak (Cores)": round(cpu_utilization_peak, 2),
            "CPU Utilization Average (Cores)": round(cpu_utilization_average, 2),
            "GPU Utilization Peak (%)": round(gpu_utilization_peak, 2),
            "GPU Utilization Average (%)": round(gpu_utilization_average, 2),
            "Start Time": start,
            "End Time": end
        })

print(f"Metrics have been written to {csv_filename}")
