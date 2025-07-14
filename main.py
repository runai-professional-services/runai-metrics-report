import datetime
import csv
import os
import concurrent.futures
from runai.configuration import Configuration
from runai.api_client import ThreadedApiClient
from runai.runai_client import RunaiClient
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import multiprocessing


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


def process_time_window(client, workload_id: str, window_start: datetime, window_end: datetime, metrics_types: list) -> Dict[str, Any]:
    """Process a single time window for a workload and return the metrics"""
    try:
        metrics_response = client.workloads.workloads.get_workload_metrics(
            workload_id=workload_id,
            start=window_start.isoformat(),
            end=window_end.isoformat(),
            metric_type=metrics_types,
            number_of_samples=1000,
        )
        return get_response_data(metrics_response)
    except Exception as e:
        print(f"Error processing window {window_start} to {window_end}: {e}")
        return {}

def process_workload(client, workload: dict, start_date: datetime, end_date: datetime, metrics_types: list, metrics_config: dict) -> Dict[str, Any]:
    """Process a single workload and return its metrics"""
    workload_id = workload.get('id')
    workload_name = workload.get('name', 'Unknown')
    print(f"\nProcessing workload: {workload_name} (ID: {workload_id})")
    
    if not workload_id:
        return {}

    # Get time windows for optimal resolution
    time_windows = get_time_windows(start_date, end_date)
    print(f"Processing {len(time_windows)} time windows")

    # Calculate optimal number of workers for time window processing
    time_window_workers = max(1, min(10, multiprocessing.cpu_count() * 2))
    
    # Initialize metrics
    metrics = {
        "gpu_hours": 0.0,
        "gpu_allocated": 0.0,
        "gpu_allocated_hours": 0.0,
        "cpu_memory_gb": 0.0,
        "memory_hours": 0.0,
        "cpu_hours": 0.0,
        "gpu_utilization_peak": 0.0,
        "gpu_utilization_avg": 0.0,
        "cpu_utilization_peak": 0.0,
        "cpu_utilization_avg": 0.0,
        "cpu_memory_peak": 0.0,
        "cpu_memory_avg": 0.0,
        "all_measurement_timestamps": []
    }

    # Process time windows in parallel with dynamic worker count
    with concurrent.futures.ThreadPoolExecutor(max_workers=time_window_workers) as executor:
        future_to_window = {
            executor.submit(
                process_time_window, 
                client, 
                workload_id, 
                window_start, 
                window_end, 
                metrics_types
            ): (window_start, window_end) 
            for window_start, window_end in time_windows
        }

        for future in concurrent.futures.as_completed(future_to_window):
            window = future_to_window[future]
            try:
                metrics_data = future.result()
                if not metrics_data.get("measurements"):
                    continue

                # Process measurements
                for measurement in metrics_data["measurements"]:
                    if not measurement.get("values"):
                        continue

                    m = Measurement(**measurement)
                    metrics["all_measurement_timestamps"].extend([
                        m.values[0]["timestamp"],
                        m.values[-1]["timestamp"]
                    ])

                    if m.type in metrics_config:
                        metric = metrics_config[m.type]
                        total_hours, peak, weighted_avg = metric.calculate(m)

                        if m.type == "CPU_USAGE_CORES":
                            metrics["cpu_hours"] += total_hours
                            metrics["cpu_utilization_peak"] = max(metrics["cpu_utilization_peak"], peak)
                            metrics["cpu_utilization_avg"] = weighted_avg
                        elif m.type == "GPU_UTILIZATION":
                            metrics["gpu_hours"] += total_hours
                            metrics["gpu_utilization_peak"] = max(metrics["gpu_utilization_peak"], peak)
                            metrics["gpu_utilization_avg"] = weighted_avg
                        elif m.type == "GPU_ALLOCATION":
                            metrics["gpu_allocated"] = max(metrics["gpu_allocated"], peak)
                        elif m.type == "CPU_MEMORY_USAGE_BYTES":
                            conversion = 1/(1024**3)
                            metrics["cpu_memory_peak"] = max(metrics["cpu_memory_peak"], peak * conversion)
                            metrics["cpu_memory_avg"] = weighted_avg * conversion
                            metrics["memory_hours"] += total_hours * conversion

            except Exception as e:
                print(f"Error processing window {window}: {e}")

    # Calculate actual duration
    if metrics["all_measurement_timestamps"]:
        actual_start = min(metrics["all_measurement_timestamps"])
        actual_end = max(metrics["all_measurement_timestamps"])
        actual_duration = (datetime.datetime.fromisoformat(actual_end) - 
                         datetime.datetime.fromisoformat(actual_start)).total_seconds() / 3600
        metrics["gpu_allocated_hours"] = actual_duration * metrics["gpu_allocated"]
        metrics["actual_start"] = actual_start
        metrics["actual_duration"] = actual_duration
    else:
        metrics["actual_duration"] = (end_date - start_date).total_seconds() / 3600
        metrics["actual_start"] = start_date.isoformat()
        metrics["gpu_allocated_hours"] = 0

    return {
        "workload": workload,
        "metrics": metrics
    }

def get_response_data(apply_result):
    """Handle ThreadedApiClient response"""
    try:
        # Unwrap the ApplyResult object
        response = apply_result.get()
        data = response.data
        # The unwrapped response should be a dictionary
        if not isinstance(data, dict):
            print(f"Unexpected response data type: {type(data)}")
            return {}
        return data
    except Exception as e:
        print(f"Error extracting data from response: {e}")
        return {}

def main():
    # Get environment variables with defaults
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    base_url = os.getenv('BASE_URL')
    output_dir = os.getenv('OUTPUT_DIR', '/mnt/data') 
    end_date=datetime.datetime.strptime(os.getenv('END_DATE'),'%d-%m-%Y').replace(tzinfo=datetime.timezone.utc) if os.getenv('END_DATE') else datetime.datetime.now(datetime.timezone.utc)
    start_date=datetime.datetime.strptime(os.getenv('START_DATE'),'%d-%m-%Y').replace(tzinfo=datetime.timezone.utc) if os.getenv('START_DATE') else end_date - datetime.timedelta(days=7)

    if not all([client_id, client_secret, base_url]):
        raise ValueError("Missing required environment variables: CLIENT_ID, CLIENT_SECRET, and BASE_URL must be set")

    # Initialize Run:AI client
    config = Configuration(
        client_id=client_id,
        client_secret=client_secret,
        runai_base_url=base_url
    )

    client = RunaiClient(ThreadedApiClient(config))

    # Define the time range
    #end_date = datetime.datetime.now(datetime.timezone.utc)
    #start_date = end_date - datetime.timedelta(days=7)
    end = end_date.isoformat()
    start = start_date.isoformat()

    print(f"Analyzing data from {start} to {end}")

    # Fetch workloads data
    try:
        response = client.workloads.workloads.get_workloads()
        response_data = get_response_data(response)
        workloads_data = response_data.get('workloads', [])
        print(f"Total Workloads Found: {len(workloads_data)}")
    except Exception as e:
        print(f"Error fetching workloads: {e}")
        return

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

    # Define filenames with full paths
    allocation_filename = os.path.join(output_dir, f"project_allocations_{start_str}_to_{end_str}.csv")
    utilization_filename = os.path.join(output_dir, f"utilization_metrics_{start_str}_to_{end_str}.csv")
    suspicious_filename = os.path.join(output_dir, f"suspicious_metrics_{start_str}_to_{end_str}.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

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

    # Calculate optimal number of workers for workload processing
    workload_workers = max(1, min(5, multiprocessing.cpu_count()))

    # Process workloads in parallel with dynamic worker count
    with concurrent.futures.ThreadPoolExecutor(max_workers=workload_workers) as executor:
        future_to_workload = {
            executor.submit(
                process_workload, 
                client, 
                workload, 
                start_date, 
                end_date, 
                metrics_types, 
                metrics_config
            ): workload 
            for workload in workloads_data
        }

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

            for future in concurrent.futures.as_completed(future_to_workload):
                workload = future_to_workload[future]
                try:
                    result = future.result()
                    if not result:
                        continue

                    workload = result["workload"]
                    metrics = result["metrics"]

                    # Write utilization metrics
                    util_writer.writerow({
                        "Project": workload.get('projectName', 'Unknown'),
                        "User": workload.get('submittedBy', 'Unknown'),
                        "Job Name": workload.get('name', 'Unknown'),
                        "GPU Hours": f"{metrics['gpu_allocated_hours']:.2f}",
                        "Memory Hours (GB)": f"{metrics['memory_hours']:.2f}",
                        "CPU Hours": f"{metrics['cpu_hours']:.2f}",
                        "GPU Allocated": f"{metrics['gpu_allocated']:.2f}",
                        "GPU Utilization % - Peak": f"{metrics['gpu_utilization_peak']:.2f}",
                        "GPU Utilization % - Average": f"{metrics['gpu_utilization_avg']:.2f}",
                        "CPU Utilization (Cores) - Peak": f"{metrics['cpu_utilization_peak']:.2f}",
                        "CPU Utilization (Cores) - Average": f"{metrics['cpu_utilization_avg']:.2f}",
                        "CPU Memory (GB) - Peak": f"{metrics['cpu_memory_peak']:.2f}",
                        "CPU Memory (GB) - Average": f"{metrics['cpu_memory_avg']:.2f}"
                    })

                    # Process project data
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
                    pd["Project Allocated GPUs"] += metrics["gpu_allocated"]
                    pd["Allocated GPU - Peak"] = max(pd["Allocated GPU - Peak"], metrics["gpu_allocated"])
                    pd["Allocated GPU - Avg"] = (pd["Allocated GPU - Avg"] * pd["count"] + metrics["gpu_allocated"]) / (pd["count"] + 1)
                    pd["CPU Memory (GB) - Peak"] = max(pd["CPU Memory (GB) - Peak"], metrics["cpu_memory_peak"])
                    pd["CPU Memory (GB) - Avg"] = (pd["CPU Memory (GB) - Avg"] * pd["count"] + metrics["cpu_memory_avg"]) / (pd["count"] + 1)
                    pd["CPU (# Cores) - Peak"] = max(pd["CPU (# Cores) - Peak"], metrics["cpu_utilization_peak"])
                    pd["CPU (# Cores) - Avg"] = (pd["CPU (# Cores) - Avg"] * pd["count"] + metrics["cpu_utilization_avg"]) / (pd["count"] + 1)
                    pd["count"] += 1

                    # Check for suspicious patterns
                    suspicious_patterns = detect_suspicious_patterns(
                        workload, metrics, metrics["actual_start"], metrics["actual_duration"]
                    )
                    for pattern in suspicious_patterns:
                        susp_writer.writerow(pattern)

                except Exception as e:
                    print(f"Error processing workload {workload.get('name', 'Unknown')}: {e}")

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
