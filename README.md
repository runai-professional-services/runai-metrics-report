# Resource Metrics Calculations

## GPU-Related Metrics

### GPU Hours
GPU Hours represent the total time GPUs were allocated to workloads, calculated as:
- GPU Hours = Number of GPUs Allocated × Duration of Allocation
- Example: If a job used 2 GPUs for 5 hours, GPU Hours = 2 × 5 = 10 hours


### GPU Utilization
GPU Utilization measures how intensively the allocated GPUs were used:
- Peak Utilization: Highest utilization percentage observed during the workload
- Average Utilization: Time-weighted average of GPU usage over the workload duration
- Calculated by sampling GPU usage at intervals and weighting by time period
- Example: If GPU was 80% utilized for 2 hours and 40% for 1 hour:
  Average = (80% × 2 + 40% × 1) ÷ 3 = 66.7%

## Memory Metrics

### Memory Hours (GB)
Memory Hours represent the total memory consumption over time:
- Calculated by converting memory usage from bytes to gigabytes
- Each measurement is multiplied by its duration in hours
- Final value is GB-Hours (memory capacity × time)
- Example: Using 2GB for 3 hours = 6 GB-Hours

### CPU Memory Usage
CPU Memory metrics track the RAM usage:
- Peak Memory: Highest memory usage observed (in GB)
- Average Memory: Time-weighted average of memory usage
- Values are converted from bytes to GB for readability
- Example: If 1.5GB used for 2 hours and 2.5GB for 1 hour:
  Average = (1.5 × 2 + 2.5 × 1) ÷ 3 = 1.83GB

## CPU Metrics

### CPU Hours
CPU Hours measure the total CPU core time consumed:
- Calculated as number of CPU cores × hours used
- Based on actual CPU core utilization
- Example: Using 2 CPU cores at 100% for 4 hours = 8 CPU Hours

### CPU Utilization
CPU Utilization tracks how intensively CPU cores were used:
- Peak Utilization: Maximum number of cores used at any point
- Average Utilization: Time-weighted average of core usage
- Example: If using 4 cores at 50% for 2 hours:
  CPU Hours = 4 × 0.5 × 2 = 4 core-hours

## Project-Level Aggregation

### Project Total Metrics
For project-level reporting:
- Total GPUs: Sum of all GPUs allocated across workloads
- Peak Values: Maximum observed across all workloads
- Average Values: Weighted average based on workload duration
- Example: If Project has 3 workloads:
  - Workload 1: 2 GPUs for 3 hours
  - Workload 2: 1 GPU for 4 hours
  - Workload 3: 3 GPUs for 2 hours
  - Project Total GPU Hours = (2×3) + (1×4) + (3×2) = 14 GPU Hours

### Time Window Considerations
- Metrics are calculated within the specified time window
- For running jobs, only the portion within the time window is counted
- Jobs that finished or started during the window are pro-rated accordingly
