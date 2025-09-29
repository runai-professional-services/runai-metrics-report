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


# Login to Run:AI
def login():  
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    base_url = os.getenv('BASE_URL')

    if not all([client_id, client_secret, base_url]):
        raise ValueError("Missing required environment variables: CLIENT_ID, CLIENT_SECRET, and BASE_URL must be set")

    # Initialize Run:AI client
    config = Configuration(
        client_id=client_id,
        client_secret=client_secret,
        runai_base_url=base_url
    )

    client = RunaiClient(ThreadedApiClient(config))
    return client


# Get time windows for optimal metric resolution
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


# Detect suspicious patterns in workload metrics
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


# Process a single time window for a workload and return the metrics
def process_time_window(client, workload_id: str, window_start: datetime, window_end: datetime, metrics_types: list) -> Dict[str, Any]:
    """Process a single time window for a workload and return the metrics"""
    try:
        # Split metrics into batches of 10 to avoid API limit
        max_metrics_per_request = 10
        all_measurements = []
        
        for i in range(0, len(metrics_types), max_metrics_per_request):
            batch_metrics = metrics_types[i:i + max_metrics_per_request]
            
            metrics_response = client.workloads.workloads.get_workload_metrics(
                workload_id=workload_id,
                start=window_start.isoformat(),
                end=window_end.isoformat(),
                metric_type=batch_metrics,
                number_of_samples=1000,
            )
            
            batch_data = get_response_data(metrics_response)
            if batch_data.get("measurements"):
                all_measurements.extend(batch_data["measurements"])
        
        # Return combined results in the same format as original
        return {"measurements": all_measurements}
        
    except Exception as e:
        print(f"Error processing window {window_start} to {window_end}: {e}")
        return {}


# Process a single workload and return its metrics
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
        "gpu_memory_request_gb": 0.0,
        "cpu_limit_cores": 0.0,
        "cpu_memory_request_gb": 0.0,
        "cpu_memory_limit_gb": 0.0,
        "pod_count": 0.0,
        "running_pod_count_peak": 0.0,
        "running_pod_count_avg": 0.0,
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
                        # New metrics processing for versions 2.20+
                        elif m.type == "GPU_MEMORY_REQUEST_BYTES":
                            # Static value - GPU memory request in GB
                            gpu_memory_gb = peak * (1/(1024**3))
                            metrics["gpu_memory_request_gb"] = max(metrics["gpu_memory_request_gb"], gpu_memory_gb)
                        elif m.type == "CPU_LIMIT_CORES":
                            # Static value - CPU limit in cores
                            metrics["cpu_limit_cores"] = max(metrics["cpu_limit_cores"], peak)
                        elif m.type == "CPU_MEMORY_REQUEST_BYTES":
                            # Static value - CPU memory request in GB
                            metrics["cpu_memory_request_gb"] = max(metrics["cpu_memory_request_gb"], peak * (1/(1024**3)))
                        elif m.type == "CPU_MEMORY_LIMIT_BYTES":
                            # Static value - CPU memory limit in GB
                            metrics["cpu_memory_limit_gb"] = max(metrics["cpu_memory_limit_gb"], peak * (1/(1024**3)))
                        elif m.type == "POD_COUNT":
                            # Static value - number of pods requested
                            metrics["pod_count"] = max(metrics["pod_count"], peak)
                        elif m.type == "RUNNING_POD_COUNT":
                            # Dynamic value - running pod count
                            metrics["running_pod_count_peak"] = max(metrics["running_pod_count_peak"], peak)
                            metrics["running_pod_count_avg"] = weighted_avg

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


# Handle ThreadedApiClient response
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


# Set the metrics types based on the cluster version
def set_metrics_types():
    """Set the metrics types based on the cluster version"""
    return [
        "GPU_ALLOCATION",
        "GPU_UTILIZATION",
        "GPU_MEMORY_USAGE_BYTES",
        "CPU_REQUEST_CORES",
        "CPU_USAGE_CORES",
        "CPU_MEMORY_USAGE_BYTES",
        "GPU_MEMORY_REQUEST_BYTES",
        "CPU_LIMIT_CORES",
        "CPU_MEMORY_REQUEST_BYTES",
        "CPU_MEMORY_LIMIT_BYTES",
        "POD_COUNT",
        "RUNNING_POD_COUNT"
    ]


# Set the metrics config based on the cluster version
def set_metrics_config():
    """Set the metrics config based on the cluster version"""
    return {
        "GPU_ALLOCATION": WorkloadMetric(type="GPU_ALLOCATION", is_static_value=True),
        "CPU_REQUEST_CORES": WorkloadMetric(type="CPU_REQUEST_CORES", is_static_value=True),
        "CPU_USAGE_CORES": WorkloadMetric(type="CPU_USAGE_CORES"),
        "GPU_UTILIZATION": WorkloadMetric(type="GPU_UTILIZATION"),
        "GPU_MEMORY_USAGE_BYTES": WorkloadMetric(type="GPU_MEMORY_USAGE_BYTES", conversion_factor=1/(1024**3)),
        "GPU_MEMORY_REQUEST_BYTES": WorkloadMetric(type="GPU_MEMORY_REQUEST_BYTES", is_static_value=True, conversion_factor=1/(1024**3)),
        "CPU_LIMIT_CORES": WorkloadMetric(type="CPU_LIMIT_CORES", is_static_value=True),
        "CPU_MEMORY_REQUEST_BYTES": WorkloadMetric(type="CPU_MEMORY_REQUEST_BYTES", is_static_value=True, conversion_factor=1/(1024**3)),
        "CPU_MEMORY_LIMIT_BYTES": WorkloadMetric(type="CPU_MEMORY_LIMIT_BYTES", is_static_value=True, conversion_factor=1/(1024**3)),
        "POD_COUNT": WorkloadMetric(type="POD_COUNT", is_static_value=True),
        "RUNNING_POD_COUNT": WorkloadMetric(type="RUNNING_POD_COUNT"),
    }


def get_time_range():
    """
    Get the time range for data collection from environment variables.
    
    Returns:
        tuple: (start_date, end_date) as datetime objects with UTC timezone
    """
    # Define the time range
    if os.getenv('END_DATE'):
        end_date = datetime.datetime.strptime(os.getenv('END_DATE'), '%d-%m-%Y').replace(tzinfo=datetime.timezone.utc)
    else:
        end_date = datetime.datetime.now(datetime.timezone.utc)
    
    if os.getenv('START_DATE'):
        start_date = datetime.datetime.strptime(os.getenv('START_DATE'), '%d-%m-%Y').replace(tzinfo=datetime.timezone.utc)
    else:
        start_date = end_date - datetime.timedelta(days=7)
    
    return start_date, end_date


def fetch_workloads_data(client):
    """
    Fetch workloads data from the Run:AI client.
    
    Args:
        client: The Run:AI client instance
        
    Returns:
        list: List of workloads data, or None if an error occurred
    """
    try:
        response = client.workloads.workloads.get_workloads()
        response_data = get_response_data(response)
        workloads_data = response_data.get('workloads', [])
        print(f"Total Workloads Found: {len(workloads_data)}")
        return workloads_data
    except Exception as e:
        print(f"Error fetching workloads: {e}")
        return None


def main():
    # Login to Run:AI
    client = login()

    # Set the output directory, defaults to /mnt/data
    output_dir = os.getenv('OUTPUT_DIR', '/mnt/data') 

    # Get the time range
    start_date, end_date = get_time_range()

    # Fetch workloads data
    workloads_data = fetch_workloads_data(client)
    if workloads_data is None:
        return

    # Format dates for filenames
    date_format = "%m-%d-%y"
    start_str = start_date.strftime(date_format)
    end_str = end_date.strftime(date_format)

    print(f"Analyzing data from {start_str} to {end_str}")

    # Metrics types for version 2.20, 2.21, 2.22
    metrics_types = set_metrics_types()

    # Metrics config for version 2.20, 2.21, 2.22
    metrics_config = set_metrics_config()

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
        "CPU (# Cores) - Avg",
        "GPU Memory Request (GB) - Peak",
        "GPU Memory Request (GB) - Avg",
        "CPU Limit (Cores) - Peak",
        "CPU Limit (Cores) - Avg",
        "CPU Memory Request (GB) - Peak",
        "CPU Memory Request (GB) - Avg",
        "CPU Memory Limit (GB) - Peak",
        "CPU Memory Limit (GB) - Avg",
        "Pod Count - Peak",
        "Pod Count - Avg",
        "Running Pod Count - Peak",
        "Running Pod Count - Avg"
    ]

    # Utilization headers
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
        "CPU Memory (GB) - Average",
        "GPU Memory Request (GB)",
        "CPU Limit (Cores)",
        "CPU Memory Request (GB)",
        "CPU Memory Limit (GB)",
        "Pod Count",
        "Running Pod Count - Peak",
        "Running Pod Count - Average"
    ]

    # Suspicious headers
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
                        "CPU Memory (GB) - Average": f"{metrics['cpu_memory_avg']:.2f}",
                        # New metrics for versions 2.20+
                        "GPU Memory Request (GB)": f"{metrics['gpu_memory_request_gb']:.2f}",
                        "CPU Limit (Cores)": f"{metrics['cpu_limit_cores']:.2f}",
                        "CPU Memory Request (GB)": f"{metrics['cpu_memory_request_gb']:.2f}",
                        "CPU Memory Limit (GB)": f"{metrics['cpu_memory_limit_gb']:.2f}",
                        "Pod Count": f"{metrics['pod_count']:.0f}",
                        "Running Pod Count - Peak": f"{metrics['running_pod_count_peak']:.0f}",
                        "Running Pod Count - Average": f"{metrics['running_pod_count_avg']:.2f}"
                    })

                    # Process project data
                    project_name = workload.get('projectName', 'Unknown')
                    if project_name not in project_data:
                        project_data[project_name] = {
                            "Department": workload.get('departmentName', 'Unknown'),
                            "Project": project_name,
                            "Project Allocated GPUs": 0,
                            "Allocated GPU - Peak": 0,
                            "Allocated GPU - Avg": 0,
                            "CPU Memory (GB) - Peak": 0,
                            "CPU Memory (GB) - Avg": 0,
                            "CPU (# Cores) - Peak": 0,
                            "CPU (# Cores) - Avg": 0,
                            "GPU Memory Request (GB) - Peak": 0,
                            "GPU Memory Request (GB) - Avg": 0,
                            "CPU Limit (Cores) - Peak": 0,
                            "CPU Limit (Cores) - Avg": 0,
                            "CPU Memory Request (GB) - Peak": 0,
                            "CPU Memory Request (GB) - Avg": 0,
                            "CPU Memory Limit (GB) - Peak": 0,
                            "CPU Memory Limit (GB) - Avg": 0,
                            "Pod Count - Peak": 0,
                            "Pod Count - Avg": 0,
                            "Running Pod Count - Peak": 0,
                            "Running Pod Count - Avg": 0,
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
                    pd["GPU Memory Request (GB) - Peak"] = max(pd["GPU Memory Request (GB) - Peak"], metrics["gpu_memory_request_gb"])
                    pd["GPU Memory Request (GB) - Avg"] = (pd["GPU Memory Request (GB) - Avg"] * pd["count"] + metrics["gpu_memory_request_gb"]) / (pd["count"] + 1)
                    pd["CPU Limit (Cores) - Peak"] = max(pd["CPU Limit (Cores) - Peak"], metrics["cpu_limit_cores"])
                    pd["CPU Limit (Cores) - Avg"] = (pd["CPU Limit (Cores) - Avg"] * pd["count"] + metrics["cpu_limit_cores"]) / (pd["count"] + 1)
                    pd["CPU Memory Request (GB) - Peak"] = max(pd["CPU Memory Request (GB) - Peak"], metrics["cpu_memory_request_gb"])
                    pd["CPU Memory Request (GB) - Avg"] = (pd["CPU Memory Request (GB) - Avg"] * pd["count"] + metrics["cpu_memory_request_gb"]) / (pd["count"] + 1)
                    pd["CPU Memory Limit (GB) - Peak"] = max(pd["CPU Memory Limit (GB) - Peak"], metrics["cpu_memory_limit_gb"])
                    pd["CPU Memory Limit (GB) - Avg"] = (pd["CPU Memory Limit (GB) - Avg"] * pd["count"] + metrics["cpu_memory_limit_gb"]) / (pd["count"] + 1)
                    pd["Pod Count - Peak"] = max(pd["Pod Count - Peak"], metrics["pod_count"])
                    pd["Pod Count - Avg"] = (pd["Pod Count - Avg"] * pd["count"] + metrics["pod_count"]) / (pd["count"] + 1)
                    pd["Running Pod Count - Peak"] = max(pd["Running Pod Count - Peak"], metrics["running_pod_count_peak"])
                    pd["Running Pod Count - Avg"] = (pd["Running Pod Count - Avg"] * pd["count"] + metrics["running_pod_count_avg"]) / (pd["count"] + 1)
                    
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
