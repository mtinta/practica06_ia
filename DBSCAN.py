import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Cargar el archivo CSV
df = pd.read_csv("/content/twitchdata-update.csv")

# Seleccionar las columnas relevantes
data = df[["Watch time(Minutes)", "Stream time(minutes)"]]

# Normalizar los datos para que tengan media cero y varianza unitaria (importante para DBSCAN)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Usar NearestNeighbors para calcular la distancia media al k-ésimo vecino más cercano
neighbors = NearestNeighbors(n_neighbors=2)
neighbors_fit = neighbors.fit(normalized_data)
distances, indices = neighbors_fit.kneighbors(normalized_data)
distances = sorted(distances[:, 1])

# Graficar la distancia al k-ésimo vecino más cercano para determinar epsilon (eps)
plt.plot(list(range(1, len(distances) + 1)), distances)
plt.title('Distancia al k-ésimo vecino más cercano')
plt.xlabel('Índice de muestra')
plt.ylabel('Distancia al k-ésimo vecino más cercano')
plt.show()

# Basado en el gráfico anterior, elige un valor para epsilon (eps)
eps_value = 0.5  # Este valor debería ajustarse según el gráfico de distancia al k-ésimo vecino

# Aplicar DBSCAN con el valor de epsilon seleccionado
dbscan = DBSCAN(eps=eps_value, min_samples=10)  # Ajusta min_samples según tu dataset
df['cluster'] = dbscan.fit_predict(normalized_data)

# Visualizar los resultados
plt.scatter(df['Watch time(Minutes)'], df['Stream time(minutes)'], c=df['cluster'], cmap='viridis')
plt.title('Resultado de DBSCAN')
plt.xlabel('Watch Time (1e9)')
plt.ylabel('Stream Time (minutos)')
plt.show()
