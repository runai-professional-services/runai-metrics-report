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
    conversion_factor: float = Field(default=1.0)

    def calculate(self, measurement: Measurement) -> tuple[float, float, float]:
        """Calculate total, peak, and weighted average for the metric"""
        total = 0.0
        peak = 0.0
        total_time = 0.0
        timestamp_prev = measurement.values[0]["timestamp"]
        
        print(f"Calculating metrics for {self.type}")
        print(f"Number of measurements: {len(measurement.values)}")

        for granular in measurement.values:
            timestamp = datetime.datetime.fromisoformat(granular["timestamp"])
            prev_timestamp = datetime.datetime.fromisoformat(str(timestamp_prev))
            time_diff = (timestamp - prev_timestamp).total_seconds()
            total_time += time_diff

            value = float(granular["value"])
            if not self.is_static_value:
                total += value * time_diff  # Keep raw sum for average calculation
            else:
                total += value * time_diff

            peak = max(peak, value)
            timestamp_prev = granular["timestamp"]

        # Convert to hours and apply conversion factor
        weighted_avg = (total / total_time) if total_time > 0 else 0
        total_hours = total / 3600 * self.conversion_factor
        
        print(f"Results for {self.type}:")
        print(f"  Total hours: {total_hours}")
        print(f"  Peak: {peak}")
        print(f"  Weighted average: {weighted_avg}")
        
        return total_hours, peak, weighted_avg


# Initialize Run:AI client
config = Configuration(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    runai_base_url=os.getenv('BASE_URL')
)

client = RunaiClient(ApiClient(config))

# Define the time range
end_date = datetime.datetime.now(datetime.timezone.utc)
start_date = end_date - datetime.timedelta(days=7)
end = end_date.isoformat()
start = start_date.isoformat()

print(f"Analyzing data from {start} to {end}")

# Format dates for filenames
date_format = "%m-%d-%y"
start_str = start_date.strftime(date_format)
end_str = end_date.strftime(date_format)

metrics_types = [
    "GPU_ALLOCATION",
    "GPU_UTILIZATION",
    "GPU_MEMORY_USAGE_BYTES",
    "CPU_REQUEST_CORES",
    "CPU_USAGE_CORES",
    "CPU_MEMORY_USAGE_BYTES"
]

metrics_config = {
    "GPU_ALLOCATION": WorkloadMetric(type="GPU_ALLOCATION", is_static_value=True),
    "CPU_REQUEST_CORES": WorkloadMetric(type="CPU_REQUEST_CORES", is_static_value=True),
    "CPU_USAGE_CORES": WorkloadMetric(type="CPU_USAGE_CORES"),
    "GPU_UTILIZATION": WorkloadMetric(type="GPU_UTILIZATION"),
    "CPU_MEMORY_USAGE_BYTES": WorkloadMetric(type="CPU_MEMORY_USAGE_BYTES", conversion_factor=1/(1024**3)),
    "GPU_MEMORY_USAGE_BYTES": WorkloadMetric(type="GPU_MEMORY_USAGE_BYTES", conversion_factor=1/(1024**3)),
}

allocation_filename = f"project_allocations_{start_str}_to_{end_str}.csv"
utilization_filename = f"utilization_metrics_{start_str}_to_{end_str}.csv"

allocation_headers = [
    "Department",
    "Project",
    "Project Allocated GPUs",
    "Allocated GPU - Peak",
    "Allocated GPU - Avg",
    "CPU Memory (GB) - Peak",
    "CPU Memory (GB) - Avg",
    "CPU (# Cores) - Peak",
    "CPU (# Cores) - Avg"
]

utilization_headers = [
    "Project",
    "User",
    "Job Name",
    "GPU Hours",
    "Memory Hours (GB)",
    "CPU Hours",
    "GPU Allocated",
    "GPU Utilization % - Peak",
    "GPU Utilization % - Average",
    "CPU Utilization (Cores) - Peak",
    "CPU Utilization (Cores) - Average",
    "CPU Memory (GB) - Peak",
    "CPU Memory (GB) - Average"
]

project_data = {}

with open(allocation_filename, 'w', newline='') as alloc_file, \
     open(utilization_filename, 'w', newline='') as util_file:
    
    alloc_writer = csv.DictWriter(alloc_file, fieldnames=allocation_headers)
    util_writer = csv.DictWriter(util_file, fieldnames=utilization_headers)
    
    alloc_writer.writeheader()
    util_writer.writeheader()

    try:
        response = client.workloads.workloads.get_workloads()
        workloads_data = response.data.get('workloads', [])
        print(f"Total Workloads Found: {len(workloads_data)}")
    except Exception as e:
        print(f"Error fetching workloads: {e}")
        exit(1)

    for workload in workloads_data:
        workload_id = workload.get('id')
        workload_name = workload.get('name', 'Unknown')
        print(f"\nProcessing workload: {workload_name} (ID: {workload_id})")
        
        if not workload_id:
            print("Skipping workload with no ID")
            continue

        try:
            metrics_response = client.workloads.workloads.get_workload_metrics(
                workload_id=workload_id,
                start=start,
                end=end,
                metric_type=metrics_types,
                number_of_samples=1000,
            )

            if not hasattr(metrics_response, 'data') or not metrics_response.data:
                print(f"No metrics data for workload {workload_name}")
                continue

            metrics = metrics_response.data
        except Exception as e:
            print(f"Error fetching metrics for {workload_name}: {e}")
            continue

        if not metrics.get("measurements"):
            print(f"No measurements available for workload {workload_name}")
            continue

        # Calculate actual job duration from measurements
        measurements = metrics["measurements"]
        measurement_timestamps = [
            timestamp 
            for m in measurements 
            if m.get("values") 
            for timestamp in [m["values"][0]["timestamp"], m["values"][-1]["timestamp"]]
        ]
        
        if measurement_timestamps:
            actual_start = min(measurement_timestamps)
            actual_end = max(measurement_timestamps)
            actual_duration = (datetime.datetime.fromisoformat(actual_end) - 
                             datetime.datetime.fromisoformat(actual_start)).total_seconds() / 3600
            print(f"Job actual duration: {actual_duration:.2f} hours")
        else:
            actual_duration = (end_date - start_date).total_seconds() / 3600
            print("Using full time window for duration")

        # Initialize metrics
        gpu_hours = 0.0
        gpu_allocated = 0.0
        gpu_allocated_hours = 0.0
        cpu_memory_gb = 0.0
        memory_hours = 0.0
        cpu_hours = 0.0
        gpu_utilization_peak = 0.0
        gpu_utilization_avg = 0.0
        cpu_utilization_peak = 0.0
        cpu_utilization_avg = 0.0
        cpu_memory_peak = 0.0
        cpu_memory_avg = 0.0

        # Process measurements
        for measurement in measurements:
            if not measurement.get("values"):
                print(f"Skipping empty measurement for {measurement.get('type')}")
                continue

            print(f"\nProcessing {measurement['type']} measurements")
            print(f"Number of values: {len(measurement['values'])}")
            if len(measurement['values']) > 0:
                print(f"First value: {measurement['values'][0]}")
                print(f"Last value: {measurement['values'][-1]}")

            m = Measurement(**measurement)
            if m.type in metrics_config:
                metric = metrics_config[m.type]
                total_hours, peak, weighted_avg = metric.calculate(m)

                if m.type == "CPU_USAGE_CORES":
                    cpu_hours = total_hours
                    cpu_utilization_peak = peak
                    cpu_utilization_avg = weighted_avg
                elif m.type == "GPU_UTILIZATION":
                    gpu_hours = total_hours
                    gpu_utilization_peak = peak
                    gpu_utilization_avg = weighted_avg
                elif m.type == "GPU_ALLOCATION":
                    gpu_allocated = peak
                    gpu_allocated_hours = actual_duration * peak  # Use actual duration for allocation
                elif m.type == "CPU_MEMORY_USAGE_BYTES":
                    conversion = 1/(1024**3)  # Bytes to GB conversion
                    cpu_memory_gb = weighted_avg * conversion
                    memory_hours = total_hours * conversion
                    cpu_memory_peak = peak * conversion
                    cpu_memory_avg = weighted_avg * conversion

        # Validate suspicious values
        if gpu_allocated > 0 and gpu_utilization_peak == 0:
            print(f"Warning: {workload_name} has GPU allocation but no utilization recorded")
            # Consider minimum utilization for allocated GPUs
            gpu_utilization_avg = max(gpu_utilization_avg, 0.01)

        print(f"\nFinal metrics for {workload_name}:")
        print(f"GPU Hours: {gpu_allocated_hours:.2f}")
        print(f"Memory Hours (GB): {memory_hours:.2f}")
        print(f"CPU Hours: {cpu_hours:.2f}")
        print(f"GPU Utilization Peak: {gpu_utilization_peak:.2f}%")
        print(f"GPU Utilization Average: {gpu_utilization_avg:.2f}%")

        def format_number(num):
            return f"{num:.2f}"

        # Write utilization metrics
        util_writer.writerow({
            "Project": workload.get('projectName', 'Unknown'),
            "User": workload.get('submittedBy', 'Unknown'),
            "Job Name": workload_name,
            "GPU Hours": format_number(gpu_allocated_hours),
            "Memory Hours (GB)": format_number(memory_hours),
            "CPU Hours": format_number(cpu_hours),
            "GPU Allocated": format_number(gpu_allocated),
            "GPU Utilization % - Peak": format_number(gpu_utilization_peak),
            "GPU Utilization % - Average": format_number(gpu_utilization_avg),
            "CPU Utilization (Cores) - Peak": format_number(cpu_utilization_peak),
            "CPU Utilization (Cores) - Average": format_number(cpu_utilization_avg),
            "CPU Memory (GB) - Peak": format_number(cpu_memory_peak),
            "CPU Memory (GB) - Average": format_number(cpu_memory_avg)
        })

        # Aggregate project data
        project_name = workload.get('projectName', 'Unknown')
        if project_name not in project_data:
            project_data[project_name] = {
                "Department": workload.get('department', 'Unknown'),
                "Project": project_name,
                "Project Allocated GPUs": 0,
                "Allocated GPU - Peak": 0,
                "Allocated GPU - Avg": 0,
                "CPU Memory (GB) - Peak": 0,
                "CPU Memory (GB) - Avg": 0,
                "CPU (# Cores) - Peak": 0,
                "CPU (# Cores) - Avg": 0,
                "count": 0
            }

        pd = project_data[project_name]
        pd["Project Allocated GPUs"] += gpu_allocated
        pd["Allocated GPU - Peak"] = max(pd["Allocated GPU - Peak"], gpu_allocated)
        pd["Allocated GPU - Avg"] = (pd["Allocated GPU - Avg"] * pd["count"] + gpu_allocated) / (pd["count"] + 1)
        pd["CPU Memory (GB) - Peak"] = max(pd["CPU Memory (GB) - Peak"], cpu_memory_peak)
        pd["CPU Memory (GB) - Avg"] = (pd["CPU Memory (GB) - Avg"] * pd["count"] + cpu_memory_avg) / (pd["count"] + 1)
        pd["CPU (# Cores) - Peak"] = max(pd["CPU (# Cores) - Peak"], cpu_utilization_peak)
        pd["CPU (# Cores) - Avg"] = (pd["CPU (# Cores) - Avg"] * pd["count"] + cpu_utilization_avg) / (pd["count"] + 1)
        pd["count"] += 1

    # Write aggregated project data
    for project_name, data in project_data.items():
        row_data = data.copy()
        del row_data["count"]
        for key in row_data:
            if isinstance(row_data[key], (int, float)):
                row_data[key] = format_number(float(row_data[key]))
        alloc_writer.writerow(row_data)

print(f"\nProject allocations ({start_str} to {end_str}) have been written to {allocation_filename}")
print(f"Utilization metrics ({start_str} to {end_str}) have been written to {utilization_filename}")
