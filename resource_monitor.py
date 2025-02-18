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
        
        return total_hours, peak, weighted_avg


def get_time_windows(start_date, end_date, desired_resolution=15):
    """
    Generate time windows for optimal metric resolution.
    Args:
        start_date: datetime object for start
        end_date: datetime object for end
        desired_resolution: desired resolution in seconds (default 15s)
    Returns:
        List of (window_start, window_end) tuples
    """
    total_duration = (end_date - start_date).total_seconds()
    samples_per_window = 1000  # Max samples per API call
    window_duration = samples_per_window * desired_resolution  # Duration covered by one API call
    
    windows = []
    current_start = start_date
    
    while current_start < end_date:
        window_end = min(current_start + datetime.timedelta(seconds=window_duration), end_date)
        windows.append((current_start, window_end))
        current_start = window_end
    
    return windows


def detect_suspicious_patterns(workload, metrics_data, actual_start, actual_duration):
    """
    Detect suspicious patterns in workload metrics.
    Returns list of suspicious patterns found.
    """
    suspicious_patterns = []
    
    gpu_allocated = metrics_data.get("gpu_allocated", 0)
    gpu_utilization_peak = metrics_data.get("gpu_utilization_peak", 0)
    gpu_utilization_avg = metrics_data.get("gpu_utilization_avg", 0)
    cpu_utilization_avg = metrics_data.get("cpu_utilization_avg", 0)
    cpu_memory_peak = metrics_data.get("cpu_memory_peak", 0)
    cpu_memory_avg = metrics_data.get("cpu_memory_avg", 0)

    # GPU allocation without utilization
    if gpu_allocated > 0 and gpu_utilization_peak == 0:
        suspicious_patterns.append({
            "Project": workload.get('projectName', 'Unknown'),
            "Job Name": workload.get('name', 'Unknown'),
            "Issue Type": "Zero GPU Utilization",
            "Description": "GPU allocated but no utilization recorded",
            "Value": f"{gpu_allocated} GPUs",
            "Timestamp": actual_start,
            "Duration (Hours)": f"{actual_duration:.2f}"
        })
    
    # High memory allocation with low utilization
    if cpu_memory_peak > 10 and cpu_memory_avg < (cpu_memory_peak * 0.1):
        suspicious_patterns.append({
            "Project": workload.get('projectName', 'Unknown'),
            "Job Name": workload.get('name', 'Unknown'),
            "Issue Type": "Low Memory Utilization",
            "Description": "High memory allocation with low average usage",
            "Value": f"Peak: {cpu_memory_peak:.2f}GB, Avg: {cpu_memory_avg:.2f}GB",
            "Timestamp": actual_start,
            "Duration (Hours)": f"{actual_duration:.2f}"
        })
    
    # Long running job with low resource utilization
    if actual_duration > 24 and (gpu_utilization_avg < 10 or cpu_utilization_avg < 10):
        suspicious_patterns.append({
            "Project": workload.get('projectName', 'Unknown'),
            "Job Name": workload.get('name', 'Unknown'),
            "Issue Type": "Low Long-term Utilization",
            "Description": "Long running job with low resource utilization",
            "Value": f"GPU Util: {gpu_utilization_avg:.2f}%, CPU Util: {cpu_utilization_avg:.2f}%",
            "Timestamp": actual_start,
            "Duration (Hours)": f"{actual_duration:.2f}"
        })
    
    return suspicious_patterns


def main():
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

    # Define filenames
    allocation_filename = f"project_allocations_{start_str}_to_{end_str}.csv"
    utilization_filename = f"utilization_metrics_{start_str}_to_{end_str}.csv"
    suspicious_filename = f"suspicious_metrics_{start_str}_to_{end_str}.csv"

    # Define headers
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

    suspicious_headers = [
        "Project",
        "Job Name",
        "Issue Type",
        "Description",
        "Value",
        "Timestamp",
        "Duration (Hours)"
    ]

    project_data = {}
    suspicious_data = []

    # Open all CSV files
    with open(allocation_filename, 'w', newline='') as alloc_file, \
         open(utilization_filename, 'w', newline='') as util_file, \
         open(suspicious_filename, 'w', newline='') as susp_file:
        
        alloc_writer = csv.DictWriter(alloc_file, fieldnames=allocation_headers)
        util_writer = csv.DictWriter(util_file, fieldnames=utilization_headers)
        susp_writer = csv.DictWriter(susp_file, fieldnames=suspicious_headers)
        
        alloc_writer.writeheader()
        util_writer.writeheader()
        susp_writer.writeheader()

        try:
            response = client.workloads.workloads.get_workloads()
            workloads_data = response.data.get('workloads', [])
            print(f"Total Workloads Found: {len(workloads_data)}")
        except Exception as e:
            print(f"Error fetching workloads: {e}")
            return

        for workload in workloads_data:
            workload_id = workload.get('id')
            workload_name = workload.get('name', 'Unknown')
            print(f"\nProcessing workload: {workload_name} (ID: {workload_id})")
            
            if not workload_id:
                continue

            # Get time windows for optimal resolution
            time_windows = get_time_windows(start_date, end_date)
            print(f"Processing {len(time_windows)} time windows")

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

            all_measurement_timestamps = []

            # Process each time window
            for window_start, window_end in time_windows:
                try:
                    metrics_response = client.workloads.workloads.get_workload_metrics(
                        workload_id=workload_id,
                        start=window_start.isoformat(),
                        end=window_end.isoformat(),
                        metric_type=metrics_types,
                        number_of_samples=1000,
                    )

                    if not hasattr(metrics_response, 'data') or not metrics_response.data:
                        continue

                    metrics = metrics_response.data
                    if not metrics.get("measurements"):
                        continue

                    # Collect timestamps for actual duration calculation
                    for measurement in metrics["measurements"]:
                        if measurement.get("values"):
                            all_measurement_timestamps.extend([
                                measurement["values"][0]["timestamp"],
                                measurement["values"][-1]["timestamp"]
                            ])

                    # Process measurements
                    for measurement in metrics["measurements"]:
                        if not measurement.get("values"):
                            continue

                        m = Measurement(**measurement)
                        if m.type in metrics_config:
                            metric = metrics_config[m.type]
                            total_hours, peak, weighted_avg = metric.calculate(m)

                            if m.type == "CPU_USAGE_CORES":
                                cpu_hours += total_hours
                                cpu_utilization_peak = max(cpu_utilization_peak, peak)
                                cpu_utilization_avg = weighted_avg
                            elif m.type == "GPU_UTILIZATION":
                                gpu_hours += total_hours
                                gpu_utilization_peak = max(gpu_utilization_peak, peak)
                                gpu_utilization_avg = weighted_avg
                            elif m.type == "GPU_ALLOCATION":
                                gpu_allocated = max(gpu_allocated, peak)
                            elif m.type == "CPU_MEMORY_USAGE_BYTES":
                                conversion = 1/(1024**3)
                                cpu_memory_peak = max(cpu_memory_peak, peak * conversion)
                                cpu_memory_avg = weighted_avg * conversion
                                memory_hours += total_hours * conversion

                except Exception as e:
                    print(f"Error processing window {window_start} to {window_end}: {e}")
                    continue

            # Calculate actual duration
            if all_measurement_timestamps:
                actual_start = min(all_measurement_timestamps)
                actual_end = max(all_measurement_timestamps)
                actual_duration = (datetime.datetime.fromisoformat(actual_end) - 
                                 datetime.datetime.fromisoformat(actual_start)).total_seconds() / 3600
                gpu_allocated_hours = actual_duration * gpu_allocated
            else:
                actual_duration = (end_date - start_date).total_seconds() / 3600
                actual_start = start
                gpu_allocated_hours = 0

            # Check for suspicious patterns
            metrics_data = {
                "gpu_allocated": gpu_allocated,
                "gpu_utilization_peak": gpu_utilization_peak,
                "gpu_utilization_avg": gpu_utilization_avg,
                "cpu_utilization_avg": cpu_utilization_avg,
                "cpu_memory_peak": cpu_memory_peak,
                "cpu_memory_avg": cpu_memory_avg
            }
            
            suspicious_patterns = detect_suspicious_patterns(
                workload, metrics_data, actual_start, actual_duration
            )
            
            for pattern in suspicious_patterns:
                susp_writer.writerow(pattern)

            # Write utilization metrics
            util_writer.writerow({
                "Project": workload.get('projectName', 'Unknown'),
                "User": workload.get('submittedBy', 'Unknown'),
                "Job Name": workload_name,
                "GPU Hours": f"{gpu_allocated_hours:.2f}",
                "Memory Hours (GB)": f"{memory_hours:.2f}",
                "CPU Hours": f"{cpu_hours:.2f}",
                "GPU Allocated": f"{gpu_allocated:.2f}",
                "GPU Utilization % - Peak": f"{gpu_utilization_peak:.2f}",
                "GPU Utilization % - Average": f"{gpu_utilization_avg:.2f}",
                "CPU Utilization (Cores) - Peak": f"{cpu_utilization_peak:.2f}",
                "CPU Utilization (Cores) - Average": f"{cpu_utilization_avg:.2f}",
                "CPU Memory (GB) - Peak": f"{cpu_memory_peak:.2f}",
                "CPU Memory (GB) - Average": f"{cpu_memory_avg:.2f}"
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
            del row_data["count"]  # Remove the counter before writing
            for key in row_data:
                if isinstance(row_data[key], (int, float)):
                    row_data[key] = f"{float(row_data[key]):.2f}"
            alloc_writer.writerow(row_data)

    print(f"\nProject allocations ({start_str} to {end_str}) have been written to {allocation_filename}")
    print(f"Utilization metrics ({start_str} to {end_str}) have been written to {utilization_filename}")
    print(f"Suspicious metrics ({start_str} to {end_str}) have been written to {suspicious_filename}")


if __name__ == "__main__":
    main()
