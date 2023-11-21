import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Cargar el archivo CSV
df = pd.read_csv("/content/twitchdata-update.csv")

#Seleccionar las columnas relevantes
data = df[["Watch time(Minutes)", "Stream time(minutes)"]]

#Normalizar los datos para que tengan media cero y varianza unitaria (importante para K-Means)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

#Determinar el número óptimo de clusters utilizando el método del codo
wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(normalized_data)
    wcss.append(kmeans.inertia_)

#Graficar el método del codo para ayudar a encontrar el número óptimo de clusters
plt.plot(range(1, 11), wcss)
plt.title('Método del Codo')
plt.xlabel('Número de Clusters')
plt.ylabel('WCSS')  # Within-Cluster-Sum-of-Squares
plt.show()

#Basado en el método del codo, elige un número de clusters (en este caso, supongamos que es 3)
num_clusters = 5

#Aplicar K-Means con el número de clusters seleccionado
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df['cluster'] = kmeans.fit_predict(normalized_data)

#Visualizar los resultados
plt.scatter(df['Watch time(Minutes)'], df['Stream time(minutes)'], c=df['cluster'], cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='*', label='Centroides')
plt.title('Resultado de K-Means')
plt.xlabel('Watch Time (1e9)')
plt.ylabel('Stream Time (minutos)')
plt.legend()
plt.show()
