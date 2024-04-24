#Import das Libs
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from pickle import dump, load

#carrega dados
colunas = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'fResp', 'grav', 'class']
    
# Obtenha o diretório atual do script
script_diretorio = os.path.dirname(os.path.abspath(__file__))

# Caminho absoluto para o arquivo
caminho_treino = os.path.join(script_diretorio, 'env_vital_signals treino.txt')
caminho_teste = os.path.join(script_diretorio, 'env_vital_signals teste.txt')

# Leia o arquivo usando o caminho absoluto
data = pd.read_csv(caminho_treino, sep=',', names=colunas)

#separa atributos e classes
data_attributes = data[['qPA', 'pulso', 'fResp']]
data_classes = data['class']

#normaliza e balancea
normalizer = MinMaxScaler()
data_attributes_normalized = normalizer.fit_transform(data_attributes)

balancer = SMOTE()
data_attributes_balanced, data_classes_balanced = balancer.fit_resample(data_attributes_normalized, data_classes)

#treinamento com Decision Tree
dt = DecisionTreeClassifier()
cv_results_dt = cross_validate(dt, data_attributes_balanced, data_classes_balanced, cv=10)

#GridSearchCV
param_grid_dt = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#busca melhores hiperparâmetros
dt_grid = GridSearchCV(dt, param_grid_dt, refit=True, verbose=1)
dt_grid.fit(data_attributes_balanced, data_classes_balanced)

# Treinar o modelo com os melhores parâmetros encontrados
dt_best_params = dt_grid.best_estimator_
dt_best_params.fit(data_attributes_balanced, data_classes_balanced)

# Avaliar o desempenho do modelo
classes_predict_dt = dt_best_params.predict(data_attributes_normalized)
report_dt = classification_report(data_classes, classes_predict_dt)
print(report_dt)

# Salvar o modelo treinado
dump(dt_best_params, open("predict_decision_tree.pkl", "wb"))

# Salvar o modelo treinado
dump(dt_grid, open("predict_decision_tree.pkl", "wb"))

#carrega dados
test_data = pd.read_csv(caminho_teste, sep=',', names=colunas)

# Separar atributos e classes da base de dados de teste
test_data_attributes = test_data[['qPA', 'pulso', 'fResp']]
test_data_classes = test_data['class']

# Normalizar os dados de teste usando o mesmo normalizador usado nos dados de treinamento
test_data_attributes_normalized = normalizer.transform(test_data_attributes)

# Avaliar o desempenho do modelo treinado na base de dados de teste
y_pred_test = dt_best_params.predict(test_data_attributes_normalized)
report_dt_test = classification_report(test_data_classes, y_pred_test)
print("Relatório de classificação na base de dados de teste:\n", report_dt_test)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Treinamento com Decision Tree usando cross-validation
dt_cv = DecisionTreeClassifier()
y_pred_cv = cross_val_predict(dt_cv, data_attributes_balanced, data_classes_balanced, cv=10)

# Calcular a matriz de confusão com os resultados do cross-validation
cm_cv = confusion_matrix(data_classes_balanced, y_pred_cv)

# Plotar a matriz de confusão do cross-validation usando seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão - Cross-Validation')
plt.xlabel('Previsto')
plt.ylabel('Verdadeiro')
plt.show()

#Classificar o novo paciente
predict_file = load(open('predict_decision_tree.pkl', 'rb'))
#print(predict_file)
#predict_pacient = predict_file.predict(test_data_attributes_normalized[[456]])
#print(predict_pacient)

# Treinar o modelo com os novos dados
predict_file.fit(data_attributes_balanced, data_classes_balanced)

def classify_vital_signals(victims):
    # Carregar o modelo de classificação treinado
    predict_file = load(open('predict_decision_tree.pkl', 'rb'))

    # Inicializar listas para armazenar os atributos e ids das vítimas
    victim_ids = []
    vital_signals = []

    # Iterar sobre as vítimas e extrair os atributos relevantes
    for victim_id, ((x, y), attributes) in victims.items():
        # Adicionar o id da vítima à lista
        victim_ids.append(victim_id)
        # Extrair os atributos 'qPA', 'pulso' e 'fResp' e adicioná-los à lista de sinais vitais
        vital_signals.append(attributes[3:])  # Ignorando os primeiros três elementos ('id', 'pSist', 'pDiast')

    # Transformar as listas em um DataFrame
    new_patient_data = pd.DataFrame(vital_signals, columns=['qPA', 'pulso', 'fResp'])

    # Normalizar os atributos usando o mesmo normalizador usado no treinamento do modelo
    new_patient_attributes_normalized = normalizer.transform(new_patient_data)

    # Prever as classes para as novas vítimas
    new_patient_predictions = predict_file.predict(new_patient_attributes_normalized)

    # Salvar as previsões em um arquivo
    with open("pred.txt", "w") as file:
        for victim_id, classe in zip(victim_ids, new_patient_predictions):
            x = 0
            y = 0
            grav = 0
            file.write(f"{victim_id}, {x}, {y}, {grav}, {classe}\n")

    print("Previsões salvas em pred.txt")