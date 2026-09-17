# High-Performance Time Series Motif Discovery & Subsequence Search
**Parallel Computing Laboratory** — Master's Degree in Computer Engineering  
Department of Information Engineering (DINFO), University of Florence (UNIFI)  
Author: **Jacopo Bruscaglioni** (`jacopo.bruscaglioni@stud.unifi.it`)  
Repository: [MIDTERM_PATTERN_RECOGNITION_2](https://github.com/Jacopo99-br/MIDTERM_PATTERN_RECOGNITION_2.git) (branch: `rep_for_CUDA`)

---

## 1. Overview
This project implements and benchmarks high-throughput subsequence similarity search and distance profile computation across multivariate industrial time series. The evaluation analyzes:
* **Memory Access Layouts:** Array of Structures (**AoS**) versus Structure of Arrays (**SoA**).
* **Parallel Architectures:** Shared-memory multi-core scaling via **OpenMP** versus SIMT offloading via **NVIDIA CUDA**.

---

## 2. Experimental Setup

The benchmarking suite is split according to hardware capabilities:
* **Google Colab (CUDA GPU Benchmarking):** Automated execution via a dedicated shell script on an **NVIDIA Tesla T4** GPU (Turing architecture, Compute Capability 7.5, 16 GB GDDR6).
* **Local Machine (OpenMP CPU Benchmarking):** Step-by-step local build and execution on a multi-core CPU across thread configurations ($P \in \{1, 2, 4, 8\}$).

### Automated Script Execution (`setup_run.sh`)

The provisioning script `setup_run.sh` sets up the workspace, downloads the dataset, compiles the project with CUDA Turing support (`sm_75`), and forwards any optional CLI arguments directly to the executable:

```bash
!chmod +x setup_run.sh && ./setup_run.sh [OPTIONS]
```

#### Execution Modes & Directives

* **Full Benchmark Suite (Default — No Arguments):**  
  Detects both CPU and GPU hardware, executing the complete OpenMP thread sweep ($P \in \{1, 2, 4, 8\}$ for both AoS and SoA), the CUDA SIMT kernels ($Q \in \{5, 50\}$, $L \in \{100, 500, 1000\}$), and the empirical block size tuning sweep:
  ```bash
  !./setup_run.sh
  ```

* **GPU Benchmark Only (`--gpu-only`):**  
  Bypasses all OpenMP multi-core CPU loops and runs exclusively the CUDA kernels and the thread-block tuning benchmark (`block_results.csv`):
  ```bash
  !./setup_run.sh --gpu-only
  ```

* **CPU Benchmark Only (`--cpu-only`):**  
  Disables CUDA execution and device memory allocations, isolating the OpenMP parallel scaling analysis across all thread counts and memory layouts (AoS vs. SoA):
  ```bash
  !./setup_run.sh --cpu-only
  ```

## 6. Visualization & Plot Generation

The Python script `src/plot_results.py` processes the benchmark CSV logs (`Search_results.csv`, `block_results.csv`, `Search_results_gpu_only.csv`) and generates high-resolution figures in `benchmarks/plots/`:

```bash
# From project root
python src/plot_results.py
```

### Generated Figures
* **`1_openmp_strong_scaling.png`:** Strong scaling speedup curve $S(P) = T(1) / T(P)$ across threads and subsequence lengths with ideal linear speedup and hyper-threading limits.
* **`2_cuda_block_tuning.png`:** Execution time versus CUDA thread block size ($32 \le \text{blockDim.x} \le 512$).
* **`3_platform_runtime_comparison.png`:** Comparative runtime across CPU Sequential (1T), OpenMP Multi-Core (8T), and CUDA Tesla T4 ($Q=50$).
* **`4_cuda_speedup_factor.png`:** GPU speedup factors normalized against sequential CPU and 8-thread OpenMP.
* **`5_aos_vs_soa_memory_layout.png`:** Memory layout access latency comparison (AoS vs. SoA) across thread counts.

---


## 7. Dataset Citation

The experimental dataset is **FaultDetectionA**, obtained from the official UEA/UCR Time Series Archive:

* **Archive Citation:**
  > A. Bagnall, J. Lines, A. Bostrom, J. Large, and E. Keogh (2017).  
  > *The UEA & UCR Time Series Classification Repository.* arXiv preprint arXiv:1602.01711.
* **Dataset URL:**  
  [https://www.timeseriesclassification.com/description.php?Dataset=FaultDetectionA](https://www.timeseriesclassification.com/description.php?Dataset=FaultDetectionA)
