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


def detect_suspicious_patterns(workload, metrics_data, actual_start, actual_duration, metrics_types=None):
    """
    Detect suspicious patterns in workload metrics using dynamic metrics.
    Returns list of suspicious patterns found.
    """
    suspicious_patterns = []
    
    # Get dynamic metric values
    gpu_allocated = metrics_data.get("gpu_allocation_allocated", 0)
    gpu_utilization_peak = metrics_data.get("gpu_utilization_peak", 0) 
    gpu_utilization_avg = metrics_data.get("gpu_utilization_avg", 0)
    cpu_utilization_avg = metrics_data.get("cpu_usage_cores_avg", 0)
    cpu_memory_peak = metrics_data.get("cpu_memory_usage_bytes_peak", 0)
    cpu_memory_avg = metrics_data.get("cpu_memory_usage_bytes_avg", 0)

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
    
    # Initialize metrics dynamically based on discovered metrics
    metrics = {"all_measurement_timestamps": []}
    
    # Initialize all discovered metrics with default values
    for metric_type in metrics_types:
        # For each metric, track hours, peak, average, and allocated values
        metrics[f"{metric_type.lower()}_hours"] = 0.0
        metrics[f"{metric_type.lower()}_peak"] = 0.0 
        metrics[f"{metric_type.lower()}_avg"] = 0.0
        metrics[f"{metric_type.lower()}_allocated"] = 0.0

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
                        
                        # Generic processing for all discovered metrics
                        metric_key = m.type.lower()
                        
                        # Update hours (for usage/utilization metrics)
                        if not metric.is_static_value:
                            metrics[f"{metric_key}_hours"] += total_hours
                        
                        # Apply conversion factor from config (especially for memory metrics)
                        converted_peak = peak * metric.conversion_factor
                        metrics[f"{metric_key}_peak"] = max(metrics[f"{metric_key}_peak"], converted_peak)
                        
                        # Update average (for dynamic metrics) or allocated (for static metrics)
                        if metric.is_static_value:
                            converted_allocated = peak * metric.conversion_factor
                            metrics[f"{metric_key}_allocated"] = max(metrics[f"{metric_key}_allocated"], converted_allocated)
                        else:
                            converted_avg = weighted_avg * metric.conversion_factor
                            metrics[f"{metric_key}_avg"] = converted_avg

            except Exception as e:
                print(f"Error processing window {window}: {e}")

    # Calculate actual duration and allocated hours for all metrics
    if metrics["all_measurement_timestamps"]:
        actual_start = min(metrics["all_measurement_timestamps"])
        actual_end = max(metrics["all_measurement_timestamps"])
        actual_duration = (datetime.datetime.fromisoformat(actual_end) - 
                         datetime.datetime.fromisoformat(actual_start)).total_seconds() / 3600
        
        # Calculate allocated hours for all static metrics
        for metric_type in metrics_types:
            metric_key = metric_type.lower()
            config = metrics_config.get(metric_type)
            if config and config.is_static_value:
                # allocated values are already converted, so just multiply by duration
                metrics[f"{metric_key}_allocated_hours"] = actual_duration * metrics[f"{metric_key}_allocated"]
        
        metrics["actual_start"] = actual_start
        metrics["actual_duration"] = actual_duration
    else:
        metrics["actual_duration"] = (end_date - start_date).total_seconds() / 3600
        metrics["actual_start"] = start_date.isoformat()
        
        # Set allocated hours to 0 for all static metrics
        for metric_type in metrics_types:
            metric_key = metric_type.lower()
            config = metrics_config.get(metric_type)
            if config and config.is_static_value:
                metrics[f"{metric_key}_allocated_hours"] = 0

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

def discover_available_metrics(client, workload_id: str, start_date: datetime, end_date: datetime, potential_metrics: list) -> tuple[list, dict]:
    """
    Dynamically discover which metrics are available for a specific workload.
    Tests each metric type to see if it returns data.
    Returns: (available_metrics_list, detailed_metrics_info)
    """
    available_metrics = []
    metrics_details = {}
    print(f"Discovering available metrics for workload {workload_id}...")
    
    # Use a longer test window to increase chances of finding data
    test_start = start_date
    test_duration = min(datetime.timedelta(hours=6), end_date - start_date)  # Use up to 6 hours
    test_end = test_start + test_duration
    
    print(f"Discovery test window: {test_start.isoformat()} to {test_end.isoformat()}")
    print(f"Testing {len(potential_metrics)} potential metrics...")
    
    for i, metric_type in enumerate(potential_metrics):
        print(f"  [{i+1}/{len(potential_metrics)}] Testing {metric_type}...", end="")
        try:
            metrics_response = client.workloads.workloads.get_workload_metrics(
                workload_id=workload_id,
                start=test_start.isoformat(),
                end=test_end.isoformat(),
                metric_type=[metric_type],
                number_of_samples=100,  # Increased sample size
            )
            
            response_data = get_response_data(metrics_response)
            measurements = response_data.get("measurements", [])
            
            # Debug: Show API response structure
            if not measurements:
                print(f" No measurements in response")
                print(f"    API Response keys: {list(response_data.keys()) if response_data else 'None'}")
            
            # Check if we got actual data for this metric
            if measurements and any(m.get("values") for m in measurements):
                available_metrics.append(metric_type)
                
                # Collect detailed information about the metric
                sample_measurement = next(m for m in measurements if m.get("values"))
                sample_values = sample_measurement.get("values", [])
                
                # Get value range for better debugging
                values = [float(v.get("value", 0)) for v in sample_values]
                min_val = min(values) if values else 0
                max_val = max(values) if values else 0
                
                metrics_details[metric_type] = {
                    "status": "available",
                    "sample_count": len(sample_values),
                    "labels": sample_measurement.get("labels", {}),
                    "sample_value": sample_values[0].get("value") if sample_values else None,
                    "value_range": f"{min_val} to {max_val}",
                    "timestamp_range": f"{sample_values[0].get('timestamp')} to {sample_values[-1].get('timestamp')}" if len(sample_values) > 1 else sample_values[0].get('timestamp') if sample_values else None
                }
                print(f" ✅ FOUND! ({len(sample_values)} samples, range: {min_val:.2f}-{max_val:.2f})")
            else:
                reason = "No measurements returned"
                if measurements:
                    empty_measurements = [m for m in measurements if not m.get("values")]
                    if empty_measurements:
                        reason = f"Found {len(measurements)} measurements but all had empty values"
                    
                metrics_details[metric_type] = {
                    "status": "no_data",
                    "reason": reason,
                    "measurement_count": len(measurements) if measurements else 0
                }
                print(f" ❌ {reason}")
                
        except Exception as e:
            metrics_details[metric_type] = {
                "status": "error",
                "error": str(e)
            }
            print(f" ⚠️  Error: {e}")
    
    return available_metrics, metrics_details

def display_metrics_discovery_report(available_metrics: list, metrics_details: dict, all_potential_metrics: list):
    """Display a comprehensive report of metrics discovery results"""
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE METRICS DISCOVERY REPORT")
    print("="*80)
    
    print(f"\n📊 SUMMARY:")
    print(f"  • Total metrics tested: {len(all_potential_metrics)}")
    print(f"  • Available metrics: {len(available_metrics)}")
    print(f"  • Unavailable metrics: {len(all_potential_metrics) - len(available_metrics)}")
    print(f"  • Success rate: {len(available_metrics)/len(all_potential_metrics)*100:.1f}%")
    
    print(f"\n✅ AVAILABLE METRICS ({len(available_metrics)}):")
    for metric in available_metrics:
        details = metrics_details.get(metric, {})
        sample_count = details.get("sample_count", 0)
        sample_value = details.get("sample_value", "N/A")
        labels = details.get("labels", {})
        labels_str = f" | Labels: {labels}" if labels else ""
        print(f"  • {metric:<25} | Samples: {sample_count:>3} | Sample Value: {sample_value}{labels_str}")
    
    unavailable = [m for m in all_potential_metrics if m not in available_metrics]
    if unavailable:
        print(f"\n❌ UNAVAILABLE METRICS ({len(unavailable)}):")
        for metric in unavailable:
            details = metrics_details.get(metric, {})
            reason = details.get("reason", details.get("error", "Unknown"))
            status = details.get("status", "unknown")
            status_symbol = "⚠️" if status == "error" else "📭"
            print(f"  {status_symbol} {metric:<25} | Reason: {reason}")
    
    print(f"\n🏷️  METRIC CATEGORIES:")
    categories = {
        "GPU Metrics": [m for m in available_metrics if "GPU" in m],
        "CPU Metrics": [m for m in available_metrics if "CPU" in m],
        "Memory Metrics": [m for m in available_metrics if "MEMORY" in m],
        "Count Metrics": [m for m in available_metrics if "COUNT" in m],
        "Allocation/Request": [m for m in available_metrics if any(x in m for x in ["ALLOCATION", "REQUEST", "LIMIT"])],
        "Utilization/Usage": [m for m in available_metrics if any(x in m for x in ["UTILIZATION", "USAGE"])]
    }
    
    for category, metrics in categories.items():
        if metrics:
            print(f"  • {category}: {len(metrics)} metrics")
            for metric in metrics:
                print(f"    - {metric}")
    
    print("="*80)

def generate_dynamic_headers(discovered_metrics: list, metrics_config: dict) -> dict:
    """Generate dynamic CSV headers based on discovered metrics"""
    
    def metric_to_friendly_name(metric_name: str) -> str:
        """Convert metric name to user-friendly column name"""
        name_mapping = {
            "GPU_UTILIZATION": "GPU Utilization %",
            "GPU_ALLOCATION": "GPU Allocated",
            "GPU_MEMORY_USAGE_BYTES": "GPU Memory (GB)",
            "GPU_MEMORY_REQUEST_BYTES": "GPU Memory Request (GB)",
            "CPU_USAGE_CORES": "CPU Usage (Cores)",
            "CPU_REQUEST_CORES": "CPU Request (Cores)", 
            "CPU_LIMIT_CORES": "CPU Limit (Cores)",
            "CPU_MEMORY_USAGE_BYTES": "CPU Memory (GB)",
            "CPU_MEMORY_REQUEST_BYTES": "CPU Memory Request (GB)",
            "CPU_MEMORY_LIMIT_BYTES": "CPU Memory Limit (GB)",
            "POD_COUNT": "Pod Count",
            "RUNNING_POD_COUNT": "Running Pod Count"
        }
        return name_mapping.get(metric_name, metric_name.replace("_", " ").title())
    
    # Base headers for all CSVs
    base_utilization_headers = ["Project", "User", "Job Name"]
    base_allocation_headers = ["Department", "Project"]
    
    # Dynamic headers based on discovered metrics
    utilization_headers = base_utilization_headers.copy()
    allocation_headers = base_allocation_headers.copy()
    
    for metric in discovered_metrics:
        config = metrics_config.get(metric)
        if not config:
            continue
            
        friendly_name = metric_to_friendly_name(metric)
        
        # For utilization CSV - add detailed columns
        if config.is_static_value:
            # Static metrics: just show the allocated/requested value
            utilization_headers.append(f"{friendly_name}")
        else:
            # Dynamic metrics: show hours (if applicable), peak, and average
            if "UTILIZATION" in metric or "USAGE" in metric:
                if "MEMORY" in metric:
                    utilization_headers.append(f"{friendly_name} Hours")
                elif "GPU" in metric or "CPU" in metric:
                    utilization_headers.append(f"{friendly_name.replace('(Cores)', '')} Hours")
                    
            utilization_headers.extend([
                f"{friendly_name} - Peak",
                f"{friendly_name} - Average"
            ])
        
        # For allocation CSV - add summary columns  
        if "GPU" in metric or "CPU" in metric or "MEMORY" in metric:
            if config.is_static_value:
                allocation_headers.extend([
                    f"{friendly_name} - Total",
                    f"{friendly_name} - Peak", 
                    f"{friendly_name} - Avg"
                ])
            else:
                allocation_headers.extend([
                    f"{friendly_name} - Peak",
                    f"{friendly_name} - Avg"
                ])
    
    # Suspicious metrics headers remain static
    suspicious_headers = [
        "Project",
        "Job Name", 
        "Issue Type",
        "Description",
        "Value",
        "Timestamp",
        "Duration (Hours)"
    ]
    
    return {
        "utilization": utilization_headers,
        "allocation": allocation_headers,
        "suspicious": suspicious_headers
    }

def generate_dynamic_csv_row(workload: dict, metrics: dict, metrics_types: list, metrics_config: dict, row_type: str) -> dict:
    """Generate dynamic CSV row data based on discovered metrics"""
    
    def metric_to_friendly_name(metric_name: str) -> str:
        """Convert metric name to user-friendly column name"""
        name_mapping = {
            "GPU_UTILIZATION": "GPU Utilization %",
            "GPU_ALLOCATION": "GPU Allocated",
            "GPU_MEMORY_USAGE_BYTES": "GPU Memory (GB)",
            "GPU_MEMORY_REQUEST_BYTES": "GPU Memory Request (GB)",
            "CPU_USAGE_CORES": "CPU Usage (Cores)",
            "CPU_REQUEST_CORES": "CPU Request (Cores)", 
            "CPU_LIMIT_CORES": "CPU Limit (Cores)",
            "CPU_MEMORY_USAGE_BYTES": "CPU Memory (GB)",
            "CPU_MEMORY_REQUEST_BYTES": "CPU Memory Request (GB)",
            "CPU_MEMORY_LIMIT_BYTES": "CPU Memory Limit (GB)",
            "POD_COUNT": "Pod Count",
            "RUNNING_POD_COUNT": "Running Pod Count"
        }
        return name_mapping.get(metric_name, metric_name.replace("_", " ").title())
    
    if row_type == "utilization":
        # Base utilization row data
        row = {
            "Project": workload.get('projectName', 'Unknown'),
            "User": workload.get('submittedBy', 'Unknown'),
            "Job Name": workload.get('name', 'Unknown')
        }
        
        # Add dynamic metric columns
        for metric_type in metrics_types:
            config = metrics_config.get(metric_type)
            if not config:
                continue
                
            metric_key = metric_type.lower()
            friendly_name = metric_to_friendly_name(metric_type)
            
            if config.is_static_value:
                # Static metrics: just show the allocated/requested value
                row[friendly_name] = f"{metrics.get(f'{metric_key}_allocated', 0):.2f}"
            else:
                # Dynamic metrics: show hours (if applicable), peak, and average
                if "UTILIZATION" in metric_type or "USAGE" in metric_type:
                    if "MEMORY" in metric_type:
                        row[f"{friendly_name} Hours"] = f"{metrics.get(f'{metric_key}_hours', 0):.2f}"
                    elif "GPU" in metric_type or "CPU" in metric_type:
                        hours_name = f"{friendly_name.replace('(Cores)', '')} Hours"
                        row[hours_name] = f"{metrics.get(f'{metric_key}_hours', 0):.2f}"
                        
                row[f"{friendly_name} - Peak"] = f"{metrics.get(f'{metric_key}_peak', 0):.2f}"
                row[f"{friendly_name} - Average"] = f"{metrics.get(f'{metric_key}_avg', 0):.2f}"
        
        return row
    
    elif row_type == "allocation":
        # Base allocation row data  
        row = {
            "Department": workload.get('departmentName', 'Unknown'),
            "Project": workload.get('projectName', 'Unknown')
        }
        
        # Add dynamic metric columns for allocation summary
        for metric_type in metrics_types:
            config = metrics_config.get(metric_type)
            if not config:
                continue
                
            # Only include GPU, CPU, and Memory metrics in allocation CSV
            if not any(keyword in metric_type for keyword in ["GPU", "CPU", "MEMORY"]):
                continue
                
            metric_key = metric_type.lower()
            friendly_name = metric_to_friendly_name(metric_type)
            
            if config.is_static_value:
                row[f"{friendly_name} - Total"] = f"{metrics.get(f'{metric_key}_allocated', 0):.2f}"
                row[f"{friendly_name} - Peak"] = f"{metrics.get(f'{metric_key}_peak', 0):.2f}"
                row[f"{friendly_name} - Avg"] = f"{metrics.get(f'{metric_key}_avg', 0):.2f}"
            else:
                row[f"{friendly_name} - Peak"] = f"{metrics.get(f'{metric_key}_peak', 0):.2f}"
                row[f"{friendly_name} - Avg"] = f"{metrics.get(f'{metric_key}_avg', 0):.2f}"
        
        return row
    
    return {}

def main():
    # Get environment variables with defaults
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    base_url = os.getenv('BASE_URL')
    output_dir = os.getenv('OUTPUT_DIR', '/mnt/data')
    
    # Handle date ranges - NOW mode is default, unless specific dates are provided
    start_date_env = os.getenv('START_DATE')
    end_date_env = os.getenv('END_DATE')
    
    if start_date_env or end_date_env:
        # Use specific dates when provided
        end_date = datetime.datetime.strptime(end_date_env, '%d-%m-%Y').replace(tzinfo=datetime.timezone.utc) if end_date_env else datetime.datetime.now(datetime.timezone.utc)
        start_date = datetime.datetime.strptime(start_date_env, '%d-%m-%Y').replace(tzinfo=datetime.timezone.utc) if start_date_env else end_date - datetime.timedelta(days=7)
        print(f"📅 Using specified date range: {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')}")
    else:
        # Default to NOW mode - use very recent data for better chances of finding active metrics
        end_date = datetime.datetime.now(datetime.timezone.utc)
        start_date = end_date - datetime.timedelta(hours=4)
        print("🔥 NOW MODE (default): Using recent 4-hour window for live workload analysis")
        print("💡 Tip: Set START_DATE and/or END_DATE environment variables for historical analysis")

    if not all([client_id, client_secret, base_url]):
        raise ValueError("Missing required environment variables: CLIENT_ID, CLIENT_SECRET, and BASE_URL must be set")

    # Initialize Run:AI client
    config = Configuration(
        client_id=client_id,
        client_secret=client_secret,
        runai_base_url=base_url
    )

    client = RunaiClient(ThreadedApiClient(config))

    # Get cluster information
    try:
        clusters_response = client.organizations.clusters.get_clusters()
        clusters_data = get_response_data(clusters_response)
        
        # Handle both list and dict responses
        if isinstance(clusters_data, list):
            clusters = clusters_data
        else:
            clusters = clusters_data.get('clusters', [])
            
        print(f"Found {len(clusters)} cluster(s):")
        for cluster in clusters:
            cluster_name = cluster.get('name', 'Unknown')
            cluster_version = cluster.get('version', 'Unknown')
            cluster_id = cluster.get('id', 'Unknown')
            print(f"  - Cluster: {cluster_name} (ID: {cluster_id}, Version: {cluster_version})")
    except Exception as e:
        print(f"Error fetching cluster information: {e}")

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

    # Complete list of available workload metrics from Run:AI API documentation
    all_available_metrics = [
        "CPU_LIMIT_CORES",
        "CPU_MEMORY_LIMIT_BYTES", 
        "CPU_MEMORY_REQUEST_BYTES",
        "CPU_MEMORY_USAGE_BYTES",
        "CPU_REQUEST_CORES",
        "CPU_USAGE_CORES",
        "GPU_ALLOCATION",
        "GPU_MEMORY_REQUEST_BYTES",
        "GPU_MEMORY_USAGE_BYTES",
        "GPU_UTILIZATION",
        "POD_COUNT",
        "RUNNING_POD_COUNT"
    ]
    
    # Currently used metrics (you can modify this to use all_available_metrics if desired)
    metrics_types = [
        "GPU_ALLOCATION",
        "GPU_UTILIZATION",
        "GPU_MEMORY_USAGE_BYTES",
        "CPU_REQUEST_CORES",
        "CPU_USAGE_CORES",
        "CPU_MEMORY_USAGE_BYTES"
    ]
    
    # Auto-discover available metrics dynamically
    if workloads_data:
        print("\n=== DISCOVERING AVAILABLE METRICS ===")
        print(f"Found {len(workloads_data)} workloads to analyze:")
        for i, wl in enumerate(workloads_data):
            wl_name = wl.get('name', 'Unknown')
            wl_id = wl.get('id', 'Unknown')
            wl_project = wl.get('projectName', 'Unknown')
            wl_status = wl.get('status', 'Unknown')
            wl_department = wl.get('departmentName', 'Unknown')
            print(f"  [{i}] {wl_name} (Project: {wl_project}, Department: {wl_department}, Status: {wl_status}, ID: {wl_id[:8]}...)")
        
        # Use the first workload to test metric availability
        sample_workload = workloads_data[0]
        sample_workload_id = sample_workload.get('id')
        sample_workload_name = sample_workload.get('name', 'Unknown')
        
        print(f"\nUsing workload '{sample_workload_name}' for metrics discovery...")
        print(f"Discovery time range: {start_date.isoformat()} to {end_date.isoformat()}")
        
        if sample_workload_id:
            discovered_metrics, metrics_details = discover_available_metrics(
                client, sample_workload_id, start_date, end_date, all_available_metrics
            )
            
            # Display comprehensive metrics discovery report
            display_metrics_discovery_report(discovered_metrics, metrics_details, all_available_metrics)
            
            # Fallback to default metrics if none were discovered
            if discovered_metrics:
                print(f"\nUsing {len(discovered_metrics)} discovered metrics for analysis.\n")
                metrics_types = discovered_metrics
            else:
                print(f"\n⚠️  WARNING: No metrics discovered from '{sample_workload_name}'!")
                print("This could mean:")
                print("  • The workload wasn't active during the discovery time window")
                print("  • Metrics are delayed or not yet available")
                print("  • The workload doesn't generate the tested metric types")
                print(f"\nTrying extended discovery with a broader time window...")
                
                # Try with a broader time window (last 24 hours)
                extended_start = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=24)
                extended_end = datetime.datetime.now(datetime.timezone.utc)
                print(f"Extended discovery range: {extended_start.isoformat()} to {extended_end.isoformat()}")
                
                discovered_metrics, metrics_details = discover_available_metrics(
                    client, sample_workload_id, extended_start, extended_end, all_available_metrics
                )
                
                if discovered_metrics:
                    print(f"\n✅ Extended discovery found {len(discovered_metrics)} metrics!")
                    metrics_types = discovered_metrics
                else:
                    print(f"\n❌ Extended discovery also found no metrics.")
                    print(f"Falling back to {len(metrics_types)} default metrics.\n")
        else:
            print("Warning: Could not discover metrics - no valid workload ID found. Using default metrics.")
    else:
        print("Warning: No workloads found. Using default metrics.")

    # Create dynamic metrics configuration based on available metrics
    def create_metrics_config(metric_types: list) -> dict:
        """Create configuration for metrics based on their types"""
        config = {}
        
        for metric_type in metric_types:
            # Determine if metric is static (allocations/limits/requests/counts)
            is_static = any(keyword in metric_type for keyword in ["ALLOCATION", "LIMIT", "REQUEST", "QUOTA", "COUNT"])
            
            # Determine if metric needs memory conversion (bytes to GB)
            needs_conversion = "MEMORY" in metric_type and "BYTES" in metric_type
            
            # Create config with appropriate settings
            if needs_conversion:
                config[metric_type] = WorkloadMetric(
                    type=metric_type, 
                    is_static_value=is_static,
                    conversion_factor=1/(1024**3)
                )
            else:
                config[metric_type] = WorkloadMetric(
                    type=metric_type,
                    is_static_value=is_static
                )
                
        return config
    
    # Final safety check to prevent empty metrics list
    if not metrics_types:
        print("⚠️  WARNING: No metrics available! Using minimal default set.")
        metrics_types = ["GPU_UTILIZATION", "CPU_USAGE_CORES"]  # Minimal set that usually works
    
    metrics_config = create_metrics_config(metrics_types)
    
    print(f"\n⚙️  METRICS CONFIGURATION:")
    print(f"Created configuration for {len(metrics_config)} metrics:")
    for metric_type, config in metrics_config.items():
        static_str = " (static)" if config.is_static_value else " (dynamic)"
        conversion_str = f" | Conversion: {config.conversion_factor}" if config.conversion_factor != 1.0 else ""
        print(f"  • {metric_type:<25}{static_str}{conversion_str}")

    # Generate dynamic headers based on discovered metrics
    print(f"\n📋 GENERATING DYNAMIC CSV HEADERS...")
    dynamic_headers = generate_dynamic_headers(metrics_types, metrics_config)
    
    allocation_headers = dynamic_headers["allocation"]
    utilization_headers = dynamic_headers["utilization"] 
    suspicious_headers = dynamic_headers["suspicious"]
    
    print(f"  • Utilization CSV: {len(utilization_headers)} columns")
    print(f"  • Allocation CSV: {len(allocation_headers)} columns")
    print(f"  • Suspicious CSV: {len(suspicious_headers)} columns")

    # Define filenames with full paths
    allocation_filename = os.path.join(output_dir, f"project_allocations_{start_str}_to_{end_str}.csv")
    utilization_filename = os.path.join(output_dir, f"utilization_metrics_{start_str}_to_{end_str}.csv")
    suspicious_filename = os.path.join(output_dir, f"suspicious_metrics_{start_str}_to_{end_str}.csv")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

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

                    # Write utilization metrics using dynamic row generation
                    util_row = generate_dynamic_csv_row(workload, metrics, metrics_types, metrics_config, "utilization")
                    util_writer.writerow(util_row)

                    # Process project data dynamically
                    project_name = workload.get('projectName', 'Unknown')
                    if project_name not in project_data:
                        # Initialize project data with base fields
                        project_data[project_name] = {
                            "Department": workload.get('departmentName', 'Unknown'),
                            "Project": project_name,
                            "count": 0
                        }
                        
                        # Initialize all allocation header fields for this project
                        for header in allocation_headers:
                            if header not in ["Department", "Project", "count"]:
                                project_data[project_name][header] = 0

                    pd = project_data[project_name]
                    
                    # Update project aggregation for all discovered metrics using proper friendly names
                    def metric_to_friendly_name_for_aggregation(metric_name: str) -> str:
                        """Convert metric name to user-friendly column name - same as in CSV generation"""
                        name_mapping = {
                            "GPU_UTILIZATION": "GPU Utilization %",
                            "GPU_ALLOCATION": "GPU Allocated",
                            "GPU_MEMORY_USAGE_BYTES": "GPU Memory (GB)",
                            "GPU_MEMORY_REQUEST_BYTES": "GPU Memory Request (GB)",
                            "CPU_USAGE_CORES": "CPU Usage (Cores)",
                            "CPU_REQUEST_CORES": "CPU Request (Cores)", 
                            "CPU_LIMIT_CORES": "CPU Limit (Cores)",
                            "CPU_MEMORY_USAGE_BYTES": "CPU Memory (GB)",
                            "CPU_MEMORY_REQUEST_BYTES": "CPU Memory Request (GB)",
                            "CPU_MEMORY_LIMIT_BYTES": "CPU Memory Limit (GB)",
                            "POD_COUNT": "Pod Count",
                            "RUNNING_POD_COUNT": "Running Pod Count"
                        }
                        return name_mapping.get(metric_name, metric_name.replace("_", " ").title())
                    
                    for metric_type in metrics_types:
                        config = metrics_config.get(metric_type)
                        if not config:
                            continue
                            
                        # Only include GPU, CPU, and Memory metrics
                        if not any(keyword in metric_type for keyword in ["GPU", "CPU", "MEMORY"]):
                            continue
                            
                        metric_key = metric_type.lower()
                        friendly_name = metric_to_friendly_name_for_aggregation(metric_type)
                        
                        # Update aggregated values based on metric type
                        if config.is_static_value:
                            # For static metrics (allocations/requests)
                            total_key = f"{friendly_name} - Total"
                            peak_key = f"{friendly_name} - Peak"
                            avg_key = f"{friendly_name} - Avg"
                            
                            if total_key in pd:
                                pd[total_key] += metrics.get(f"{metric_key}_allocated", 0)
                            if peak_key in pd:
                                pd[peak_key] = max(pd[peak_key], metrics.get(f"{metric_key}_peak", 0))
                            if avg_key in pd:
                                current_avg = pd[avg_key] 
                                new_value = metrics.get(f"{metric_key}_avg", 0)
                                pd[avg_key] = (current_avg * pd["count"] + new_value) / (pd["count"] + 1)
                        else:
                            # For dynamic metrics (usage/utilization)
                            peak_key = f"{friendly_name} - Peak"
                            avg_key = f"{friendly_name} - Avg"
                            
                            if peak_key in pd:
                                pd[peak_key] = max(pd[peak_key], metrics.get(f"{metric_key}_peak", 0))
                            if avg_key in pd:
                                current_avg = pd[avg_key]
                                new_value = metrics.get(f"{metric_key}_avg", 0)
                                pd[avg_key] = (current_avg * pd["count"] + new_value) / (pd["count"] + 1)
                    
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
                
                # Format numeric values
                for key in row_data:
                    if isinstance(row_data[key], (int, float)):
                        row_data[key] = f"{float(row_data[key]):.2f}"
                
                # Ensure all allocation headers are present
                complete_row = {}
                for header in allocation_headers:
                    complete_row[header] = row_data.get(header, "0.00")
                
                alloc_writer.writerow(complete_row)

    print(f"\nProject allocations ({start_str} to {end_str}) have been written to {allocation_filename}")
    print(f"Utilization metrics ({start_str} to {end_str}) have been written to {utilization_filename}")
    print(f"Suspicious metrics ({start_str} to {end_str}) have been written to {suspicious_filename}")


if __name__ == "__main__":
    main()
