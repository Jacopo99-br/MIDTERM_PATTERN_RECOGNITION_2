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

# Creiamo una figura con due sottografici (Subplots)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# GRAFICO 1: Tempo di esecuzione Multi-Query al variare del numero di query
# Filtriamo per una lunghezza specifica, ad esempio 500
df_500 = df[(df['Type'] == 'Multi') & (df['QueryLength'] == 500)]
sns.lineplot(data=df_500, x='NumQueries', y='TimeMS', hue='Format', marker='o', ax=ax1)
ax1.set_title('Multi-Query Performance (Lunghezza: 500)')
ax1.set_ylabel('Tempo (ms)')
ax1.set_xlabel('Numero di Query')

# GRAFICO 2: Impatto della Lunghezza Query (con NumQueries fissato a 50)
df_q50 = df[(df['Type'] == 'Multi') & (df['NumQueries'] == 50)]
sns.barplot(data=df_q50, x='QueryLength', y='TimeMS', hue='Format', ax=ax2)
ax2.set_title('Impatto Lunghezza Query (50 Query totali)')
ax2.set_ylabel('Tempo (ms)')
ax2.set_xlabel('Lunghezza Query')

plt.tight_layout()
plt.savefig('benchmark_plots.png')
print("Grafici salvati con successo in 'benchmark_plots.png'")
plt.show()