import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurazione stile tipografico ad alta risoluzione per report LaTeX
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300
})

# 1. CARICAMENTO DEI DATI
csv_cpu_path = 'Search_results.csv'
csv_gpu_path = 'Search_results_gpu_only.csv'

if not os.path.exists(csv_cpu_path):
    # Prova dentro la cartella src se eseguito dalla radice
    csv_cpu_path = os.path.join('src', 'Search_results.csv')

df_cpu = pd.read_csv(csv_cpu_path)

# Rimuove eventuali run duplicate preservando l'ultima esecuzione
df_cpu = df_cpu.drop_duplicates(
    subset=['Platform', 'Format', 'QueryLength', 'NumQueries', 'Threads'], 
    keep='last'
).copy()

df_cpu['Threads'] = pd.to_numeric(df_cpu['Threads'], errors='coerce')
df_cpu['TimeS'] = df_cpu['Mean_MS'] / 1000.0
df_cpu['StdDevS'] = df_cpu['StdDev_MS'] / 1000.0

# Calcolo dello Speedup ed Efficienza OpenMP rispetto alla baseline (Threads == 1)
baseline = df_cpu[df_cpu['Threads'] == 1][['Format', 'QueryLength', 'NumQueries', 'Mean_MS']]
baseline = baseline.rename(columns={'Mean_MS': 'Base_Mean_MS'})
df_cpu = df_cpu.merge(baseline, on=['Format', 'QueryLength', 'NumQueries'])
df_cpu['Speedup'] = df_cpu['Base_Mean_MS'] / df_cpu['Mean_MS']
df_cpu['Efficiency'] = df_cpu['Speedup'] / df_cpu['Threads']

print(f"Dataset CPU caricato: {len(df_cpu)} record.")

# ==============================================================================
# FIGURA 1: TEMPI DI ESECUZIONE OPENMP (AoS vs SoA)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
batch_sizes = sorted(df_cpu['NumQueries'].unique())

for idx, q_num in enumerate(batch_sizes):
    ax = axes[idx]
    sub = df_cpu[df_cpu['NumQueries'] == q_num]
    
    sns.barplot(
        data=sub,
        x='Threads',
        y='TimeS',
        hue='Format',
        palette={'AoS': '#4C72B0', 'SoA': '#55A868'},
        ci=None,
        ax=ax
    )
    ax.set_title(f'OpenMP Execution Time (Batch Q = {q_num})', fontweight='bold')
    ax.set_xlabel('Thread Count (P)')
    ax.set_ylabel('Execution Time [s]')
    ax.legend(title='Memory Layout')

plt.tight_layout()
plt.savefig('fig1_openmp_execution_time.png')
plt.close()
print("-> Generata: fig1_openmp_execution_time.png")

# ==============================================================================
# FIGURA 2: SPEEDUP SCALABILITY (Curve di Speedup vs Lineare Ideale)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for idx, q_num in enumerate(batch_sizes):
    ax = axes[idx]
    sub = df_cpu[df_cpu['NumQueries'] == q_num]
    
    for fmt, color in [('AoS', '#4C72B0'), ('SoA', '#55A868')]:
        sub_fmt = sub[sub['Format'] == fmt]
        # Media tra le lunghezze di query per mostrare il trend medio
        trend = sub_fmt.groupby('Threads')['Speedup'].mean().reset_index()
        ax.plot(trend['Threads'], trend['Speedup'], marker='o', linewidth=2.2, 
                label=f'{fmt} (Mean)', color=color)
    
    # Riferimento Lineare Ideale
    max_t = int(df_cpu['Threads'].max())
    ax.plot([1, max_t], [1, max_t], 'k--', alpha=0.7, label='Ideal Linear')
    
    ax.set_title(f'Speedup Scaling (Q = {q_num})', fontweight='bold')
    ax.set_xlabel('Thread Count (P)')
    ax.set_ylabel('Speedup (S = T1 / Tp)')
    ax.set_xticks(sorted(df_cpu['Threads'].unique()))
    ax.legend()

plt.tight_layout()
plt.savefig('fig2_openmp_speedup.png')
plt.close()
print("-> Generata: fig2_openmp_speedup.png")

# ==============================================================================
# FIGURA 3: CONFRONTO GLOBALE SEQUENZIALE vs OPENMP (8 THREADS) vs CUDA (GPU)
# ==============================================================================
if os.path.exists(csv_gpu_path) or os.path.exists(os.path.join('src', csv_gpu_path)):
    path_g = csv_gpu_path if os.path.exists(csv_gpu_path) else os.path.join('src', csv_gpu_path)
    df_gpu = pd.read_csv(path_g)
    df_gpu['TimeS'] = df_gpu['Mean_MS'] / 1000.0
    
    # Estraiamo: 1 thread CPU (Seriale), 8 thread CPU (OpenMP), e GPU CUDA
    seq = df_cpu[(df_cpu['Threads'] == 1) & (df_cpu['Format'] == 'AoS')].copy()
    seq['Version'] = 'Sequential (AoS)'
    
    omp = df_cpu[(df_cpu['Threads'] == 8) & (df_cpu['Format'] == 'SoA')].copy()
    omp['Version'] = 'OpenMP 8-th (SoA)'
    
    cuda = df_gpu.copy()
    cuda['Version'] = 'CUDA Tesla T4 (SoA)'
    
    comp = pd.concat([
        seq[['QueryLength', 'NumQueries', 'TimeS', 'Version']],
        omp[['QueryLength', 'NumQueries', 'TimeS', 'Version']],
        cuda[['QueryLength', 'NumQueries', 'TimeS', 'Version']]
    ])
    
    plt.figure(figsize=(10, 5.5))
    g = sns.barplot(
        data=comp[comp['NumQueries'] == 50],
        x='QueryLength',
        y='TimeS',
        hue='Version',
        palette=['#C44E52', '#4C72B0', '#55A868']
    )
    plt.yscale('log')
    plt.title('Global Performance Comparison (Batch Q = 50, Log Scale)', fontweight='bold')
    plt.xlabel('Query Length (L)')
    plt.ylabel('Execution Time [s] (Logarithmic Scale)')
    plt.legend(title='Architecture')
    
    # Aggiunta etichette sui tempi
    for p in g.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            g.annotate(f'{height:.2f}s',
                       (p.get_x() + p.get_width() / 2., height),
                       ha='center', va='bottom', fontsize=8,
                       xytext=(0, 2), textcoords='offset points')
                       
    plt.tight_layout()
    plt.savefig('fig3_global_comparison_cuda.png')
    plt.close()
    print("-> Generata: fig3_global_comparison_cuda.png")