import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. LOCALIZZAZIONE DEI FILE CSV E CARTELLA PLOTS
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

candidates_dir = [
    PROJECT_ROOT / "benchmarks",
    PROJECT_ROOT / "benchmark",
    Path.cwd() / "benchmarks",
    Path.cwd() / "benchmark",
    Path.cwd()
]

BENCHMARK_DIR = next((d for d in candidates_dir if (d / "Search_results.csv").exists()), None)
if BENCHMARK_DIR is None:
    raise FileNotFoundError("Cartella benchmarks non trovata. Assicurati che Search_results.csv sia presente.")

PLOTS_DIR = BENCHMARK_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

csv_search_path = BENCHMARK_DIR / "Search_results.csv"
csv_block_path = BENCHMARK_DIR / "block_results.csv"
csv_gpu_path = BENCHMARK_DIR / "Search_results_gpu_only.csv"

# ==============================================================================
# 2. CARICAMENTO E PREPARAZIONE DATI
# ==============================================================================
df_search = pd.read_csv(csv_search_path)
df_search['Threads'] = pd.to_numeric(df_search['Threads'], errors='coerce')

# Aggregazione con media (gestisce sia run duplicate che multiple sessioni)
df_cpu = df_search.groupby(['Platform', 'Format', 'QueryLength', 'NumQueries', 'Threads'])['Mean_MS'].mean().reset_index()
df_cpu['TimeS'] = df_cpu['Mean_MS'] / 1000.0

# Calcolo baseline a 1 thread per lo speedup
base = df_cpu[df_cpu['Threads'] == 1][['Format', 'QueryLength', 'NumQueries', 'TimeS']].rename(
    columns={'TimeS': 'Base_TimeS'}
)
df_cpu = df_cpu.merge(base, on=['Format', 'QueryLength', 'NumQueries'])
df_cpu['Speedup'] = df_cpu['Base_TimeS'] / df_cpu['TimeS']

# Caricamento Tuning GPU e Risultati GPU
df_block = pd.read_csv(csv_block_path) if csv_block_path.exists() else None
df_gpu = pd.read_csv(csv_gpu_path) if csv_gpu_path.exists() else None

# Stile grafico uniforme
NAVY = '#003366'
TEAL = '#007A78'
CORAL = '#D9534F'
SLATE = '#546E7A'
AMBER = '#E69500'
BG_COLOR = '#F4F4F4'

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#000000'
plt.rcParams['axes.linewidth'] = 1.0

# ==============================================================================
# FIGURA 1: OpenMP Strong Scaling (Speedup vs. Thread Count)
# ==============================================================================
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
ax.set_facecolor(BG_COLOR)

ax.plot([1, 8], [1, 8], color='#444444', linestyle='--', linewidth=1.8, label='Ideal Linear Speedup')

colors_m = {100: NAVY, 500: TEAL, 1000: CORAL}
markers_m = {100: 'o', 500: 's', 1000: '^'}

sub_q50 = df_cpu[(df_cpu['NumQueries'] == 50) & (df_cpu['Format'] == 'SoA')]

for m_len in [100, 500, 1000]:
    data_m = sub_q50[sub_q50['QueryLength'] == m_len].sort_values('Threads')
    if not data_m.empty:
        threads = data_m['Threads'].values
        speedups = data_m['Speedup'].values
        ax.plot(threads, speedups, marker=markers_m[m_len], markersize=8, linewidth=2.5,
                color=colors_m[m_len], label=f'M = {m_len} (Q=50)')
        
        for t, s in zip(threads, speedups):
            ax.annotate(f'{s:.2f}x', (t, s), textcoords="offset points", xytext=(0, 6),
                        ha='center', fontsize=9, fontweight='bold', color=colors_m[m_len])

ax.axvspan(4, 8, color='#E0E0E0', alpha=0.5, label='Hyper-Threading Zone (contention)')

ax.set_title("OpenMP Strong Scaling (Speedup vs. Thread Count)", fontsize=13, fontweight='bold', color=NAVY, pad=15)
ax.set_xlabel("Number of Threads (p)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_ylabel("Speedup S(p) = T(1) / T(p)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_xticks([1, 2, 4, 8])
ax.set_yticks(range(1, 10))
ax.set_ylim(0.7, 9)
ax.grid(True, linestyle='--', alpha=0.6, color='#CCCCCC')
ax.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig(PLOTS_DIR / '1_openmp_strong_scaling.png')
plt.close()
print("-> Generata: plots/1_openmp_strong_scaling.png")

# ==============================================================================
# FIGURA 2: CUDA Kernel Block Size Tuning (da block_results.csv)
# ==============================================================================
if df_block is not None:
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax.set_facecolor('#FFFFFF')
    
    df_block_sorted = df_block.sort_values('BlockDim')
    blocks = df_block_sorted['BlockDim'].values
    times_ms = df_block_sorted['Mean_MS'].values
    
    ax.plot(blocks, times_ms, marker='o', markersize=8, linewidth=2.5, color=NAVY)
    
    for b, t in zip(blocks, times_ms):
        ax.annotate(f'{t:.1f} ms', (b, t), textcoords="offset points", xytext=(0, 8),
                    ha='center', fontsize=9, fontweight='bold', color=NAVY)
                    
    ax.set_title("CUDA Kernel Block Size Tuning (Fixed: L=5120, Q=50, M=500)", 
                 fontsize=13, fontweight='bold', color=NAVY, pad=15)
    ax.set_xlabel("Threads per Block (blockDim.x)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel("Execution Time (ms)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xticks(blocks)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '2_cuda_block_tuning.png')
    plt.close()
    print("-> Generata: plots/2_cuda_block_tuning.png")

# ==============================================================================
# FIGURA 3 & 4: CONFRONTO GLOBALE E SPEEDUP GPU (Q = 50 Queries)
# ==============================================================================
if df_gpu is not None:
    df_gpu['TimeS'] = df_gpu['Mean_MS'] / 1000.0
    lengths = [100, 500, 1000]
    
    seq_times, omp_times, gpu_times = [], [], []
    for l in lengths:
        t_seq = df_cpu[(df_cpu['NumQueries'] == 50) & (df_cpu['QueryLength'] == l) & 
                       (df_cpu['Threads'] == 1) & (df_cpu['Format'] == 'SoA')]['TimeS'].values[0]
        t_omp = df_cpu[(df_cpu['NumQueries'] == 50) & (df_cpu['QueryLength'] == l) & 
                       (df_cpu['Threads'] == 8) & (df_cpu['Format'] == 'SoA')]['TimeS'].values[0]
        t_gpu = df_gpu[(df_gpu['NumQueries'] == 50) & (df_gpu['QueryLength'] == l)]['TimeS'].values[0]
        
        seq_times.append(t_seq)
        omp_times.append(t_omp)
        gpu_times.append(t_gpu)

    # --- FIGURA 3: Cross-Platform Execution Time ---
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    x = np.arange(len(lengths))
    width = 0.25

    r1 = ax.bar(x - width, seq_times, width, label='CPU Sequential (1 Thread)', color=SLATE)
    r2 = ax.bar(x, omp_times, width, label='OpenMP Multi-Core (8 Threads)', color=TEAL)
    r3 = ax.bar(x + width, gpu_times, width, label='CUDA GPU (Tesla T4)', color=NAVY)

    for rects, color in [(r1, '#333333'), (r2, TEAL), (r3, NAVY)]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                        fontsize=8, fontweight='bold', color=color)

    ax.set_title("Cross-Platform Execution Time (Batch Q = 50 Queries)", fontsize=13, fontweight='bold', color=NAVY, pad=15)
    ax.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M = {l}" for l in lengths], fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '3_platform_runtime_comparison.png')
    plt.close()
    print("-> Generata: plots/3_platform_runtime_comparison.png")

    # --- FIGURA 4: CUDA GPU Speedup Factor ---
    sp_vs_seq = [s / g for s, g in zip(seq_times, gpu_times)]
    sp_vs_omp = [o / g for o, g in zip(omp_times, gpu_times)]

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    w_bar = 0.30

    rb1 = ax.bar(x - w_bar/2, sp_vs_seq, w_bar, label='GPU Speedup vs. CPU Sequential (1T)', color=AMBER)
    rb2 = ax.bar(x + w_bar/2, sp_vs_omp, w_bar, label='GPU Speedup vs. OpenMP (8 Threads)', color=NAVY)

    for rect in rb1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}x', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='#995C00')

    for rect in rb2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}x', xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=NAVY)

    ax.set_title("CUDA GPU Speedup Factor (Q = 50 Queries)", fontsize=13, fontweight='bold', color=NAVY, pad=15)
    ax.set_ylabel("Speedup Factor", fontsize=11, fontweight='bold', labelpad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M = {l}" for l in lengths], fontsize=11, fontweight='bold')
    ax.legend(loc='center right', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / '4_cuda_speedup_factor.png')
    plt.close()
    print("-> Generata: plots/4_cuda_speedup_factor.png")

# ==============================================================================
# FIGURA 5: Memory Layout Impact: AoS vs. SoA (M = 500, Q = 50)
# ==============================================================================
sub_mem = df_cpu[(df_cpu['QueryLength'] == 500) & (df_cpu['NumQueries'] == 50)]

threads_arr = [1, 2, 4, 8]
aos_times = [sub_mem[(sub_mem['Format'] == 'AoS') & (sub_mem['Threads'] == t)]['TimeS'].values[0] for t in threads_arr]
soa_times = [sub_mem[(sub_mem['Format'] == 'SoA') & (sub_mem['Threads'] == t)]['TimeS'].values[0] for t in threads_arr]

fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
x_mem = np.arange(len(threads_arr))
w_mem = 0.35

r_aos = ax.bar(x_mem - w_mem/2, aos_times, w_mem, label='AoS (Array of Structures)', color=CORAL)
r_soa = ax.bar(x_mem + w_mem/2, soa_times, w_mem, label='SoA (Structure of Arrays)', color=NAVY)

for rect in r_aos:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=CORAL)

for rect in r_soa:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}s', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color=NAVY)

ax.set_title("Memory Layout Impact: AoS vs. SoA (M = 500, Q = 50)", fontsize=13, fontweight='bold', color=NAVY, pad=15)
ax.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight='bold', labelpad=8)
ax.set_xlabel("OpenMP Threads", fontsize=11, fontweight='bold', labelpad=8)
ax.set_xticks(x_mem)
ax.set_xticklabels([f"{t} Threads" for t in threads_arr], fontsize=10)
ax.legend(loc='upper right', frameon=True)
ax.grid(True, linestyle='--', alpha=0.5, axis='y')

plt.tight_layout()
plt.savefig(PLOTS_DIR / '5_aos_vs_soa_memory_layout.png')
plt.close()
print("-> Generata: plots/5_aos_vs_soa_memory_layout.png")