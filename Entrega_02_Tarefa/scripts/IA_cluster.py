from sklearn.mixture import GaussianMixture
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def clusterize_victims(victims):

    # Obtenha o diretório atual do script
    script_diretorio = os.path.dirname(os.path.abspath(__file__))

    # Caminho completo para o arquivo de classes preditas
    caminho_pred = os.path.join(script_diretorio, '..', 'pred.txt')

    # Inicializar listas para armazenar os IDs, coordenadas e classes das vítimas
    victim_ids = []
    victim_coordinates = []
    victim_classes = []

    # Leitura do arquivo predito
    pred = pd.read_csv(caminho_pred, delimiter=',', header=None)
    pred_ids = pred.iloc[:, 0]  # IDs das vítimas preditas
    pred_classes = pred.iloc[:, -1]  # Classes preditas

    # Iterar sobre as vítimas e extrair os IDs, coordenadas (x, y) e classes
    for victim_id, ((x, y), _) in victims.items():
        # Adicionar o ID da vítima à lista
        victim_ids.append(victim_id)
        # Adicionar as coordenadas (x, y) à lista de coordenadas
        victim_coordinates.append((x, y))
        # Obter a classe prevista da vítima a partir do arquivo predito
        class_index = pred_ids[pred_ids == victim_id].index[0]
        victim_classes.append(pred_classes[class_index])

    # Criar um DataFrame com as coordenadas e classes das vítimas
    victim_data = pd.DataFrame({'ID': victim_ids, 'X': [coord[0] for coord in victim_coordinates], 'Y': [coord[1] for coord in victim_coordinates], 'Class': victim_classes})

    # Criar o objeto GMM
    num_clusters = 4
    gmm = GaussianMixture(n_components=num_clusters)
    
    # Ajustar o modelo aos dados
    gmm.fit(victim_data[['X', 'Y']])

    # Prever os rótulos dos clusters para cada amostra
    labels = gmm.predict(victim_data[['X', 'Y']])

    # Salvar os dados de cada cluster em arquivos txt
    for cluster_label in range(num_clusters):
        cluster_filename = f"cluster{cluster_label + 1}.txt"  # Cluster começa em 1
        cluster_data = victim_data[labels == cluster_label]
        with open(cluster_filename, "w") as file:
            for idx, row in cluster_data.iterrows():
                file.write(f"{row['ID']}, {row['X']}, {row['Y']}, 0, {row['Class']}\n")

    print("Clusters salvos em arquivos txt.")

    # Retorna os clusters como DataFrames
    clusters = [victim_data[labels == cluster_label] for cluster_label in range(num_clusters)]

    # Visualização dos clusters
    plt.figure(figsize=(10, 8))

    # Cores para os clusters
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    # Plotar cada cluster com uma cor diferente
    for cluster_label, color in zip(range(num_clusters), colors):
        cluster_data = victim_data[labels == cluster_label]
        plt.scatter(cluster_data['X'], cluster_data['Y'], label=f'Cluster {cluster_label + 1}', c=color)

    plt.title('Clusters')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

    return clusters