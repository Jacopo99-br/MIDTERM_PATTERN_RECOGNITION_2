import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica i dati dal CSV
try:
    df = pd.read_csv('src/Search_results.csv')
except FileNotFoundError:
    print("Errore: Il file Search_results.csv non è stato trovato.")
    exit()

# Imposta lo stile estetico
sns.set_theme(style="whitegrid")

df['TimeS'] = df['TimeMS'] / 1000.0  # Converti ms in secondi

query_lengths = sorted(df['QueryLength'].unique())
num_queries = sorted(df['NumQueries'].unique())

df_single = df[df['Type'] == 'Single']

##  FIG 1 PER SINGLE QUERY

plt.figure(figsize=(12, 6))
sns.barplot(x='QueryLength', y='TimeS', data=df_single, hue='Format')
plt.title('Single Query Performance: AoS vs SoA')
plt.xlabel('Query Length')
plt.ylabel('Time (s)')
plt.savefig('single_query_performance.png')
plt.close()


## FIG 2 PER MULTIPLE QUERIES
df_multi= df[df['Type'] == 'Multi']

fig, axes = plt.subplots(len(num_queries), 1, figsize=(12, 6 * len(num_queries)), sharex=True)

for i, num_q in enumerate(num_queries):
    ax = axes[i]
    subset = df_multi[df_multi['NumQueries'] == num_q]
    sns.barplot(data=subset, x='QueryLength', y='TimeS', hue='Format', ax=ax)
    ax.set_title(f'Multiple Queries Performance: {num_q} Queries')
    ax.set_xlabel('Query Length')
    ax.set_ylabel('Time (s)')

plt.tight_layout()
plt.savefig('multiple_queries_performance.png')
plt.close()