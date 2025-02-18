# Run:AI Metrics Collection Script

A Python script (`metrics-consumption.py`) for collecting and analyzing resource utilization metrics from Run:AI workloads. The script generates detailed reports about GPU, CPU, and memory usage across projects and workloads.

## Features

- Collects metrics with 15-second resolution (Prometheus default)
- Generates three types of reports:
  - Project Allocations
  - Utilization Metrics
  - Suspicious Usage Patterns
- Supports time-window based collection for optimal data resolution
- Handles large time ranges efficiently

## Metrics Calculated

### GPU Metrics
- GPU Hours = Number of GPUs Allocated × Duration of Allocation
- GPU Utilization (Peak and Average)
- GPU Allocation patterns

### Memory Metrics
- Memory Hours (GB) = Memory Usage × Duration
- CPU Memory Usage (Peak and Average)
- Memory utilization patterns

### CPU Metrics
- CPU Hours = Number of CPU Cores × Hours Used
- CPU Utilization (Peak and Average)
- Core usage patterns

## Prerequisites

- Python 3.12
- Run:AI API access
- Required Python packages:
  ```
  runai
  pydantic
  ```

## Environment Variables

Set the following environment variables before running:
```bash
export CLIENT_ID="your_client_id"
export CLIENT_SECRET="your_client_secret"
export BASE_URL="your_base_url"
```

## Output Files

The script generates three CSV files:

1. `project_allocations_[date_range].csv`
   - Project-level resource allocation summary
   - Includes department, GPU, CPU, and memory metrics

2. `utilization_metrics_[date_range].csv`
   - Detailed workload-level metrics
   - Includes GPU hours, memory hours, CPU hours, etc.

3. `suspicious_metrics_[date_range].csv`
   - Identifies potential resource utilization issues
   - Highlights underutilized resources and inefficient patterns

## Calculation Methods

### Time Windows
- Uses 15-second resolution for data collection
- Splits long time ranges into manageable windows
- Each window contains up to 1000 samples

### Resource Calculations
- GPU Hours: Calculated based on allocation and duration
- Memory Hours: Tracks GB-hours of memory usage
- CPU Hours: Measures actual core utilization time

### Suspicious Pattern Detection
Identifies:
- GPU allocation without utilization
- High memory allocation with low usage
- Long-running jobs with low resource utilization
