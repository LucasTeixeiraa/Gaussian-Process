import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

# Carregar os dados de um diretório
directory = "C:/Users\Lucas Teixeira\AP-Emulators\Generated_Data_100K\ModelA"  # Substitua pelo seu diretório
Ns = 10000  # Número máximo de amostras a carregar
X = pd.read_csv(os.path.join(directory, "X.csv")).iloc[0:Ns]
Y = pd.read_csv(os.path.join(directory, "Y.csv")).iloc[0:Ns]
print(f"Carregado X com forma: {X.shape}")
print(f"Carregado Y com forma: {Y.shape}")

# Dividir os dados em treino e validação
test_size = 0.5
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=42)
print(f"Treino X: {X_train.shape}, Validação X: {X_val.shape}")
print(f"Treino Y: {Y_train.shape}, Validação Y: {Y_val.shape}")

# Definir modelos
models = {
    # "Regressão Linear": LinearRegression(),
    "Processo Gaussiano": GaussianProcessRegressor(),
    # "Rede Neural": MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42),
}

# Treinar, avaliar e plotar para cada modelo
results = []
for name, model in models.items():
    print(f"Treinando {name}...")
    
    # Medir tempo de treinamento
    start_time = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - start_time
    
    # Medir tempo de inferência
    start_time = time.time()
    predictions = model.predict(X_val)
    inference_time = time.time() - start_time
    
    # Calcular MSE
    mse = mean_squared_error(Y_val, predictions)
    
    print(f"{name} MSE: {mse}, Tempo de Treino: {train_time:.2f}s, Tempo de Inferência: {inference_time:.2f}s")
    results.append({
        "Model": name,
        "MSE": mse,
        "Train Time (s)": train_time,
        "Inference Time (s)": inference_time
    })
    
    # Plotar Y verdadeiro vs Y previsto para cada QoI
    # Configurar layout do mega plot
    num_qois = len(Y.columns)
    rows = 3
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()

    for i, qoi in enumerate(Y.columns):
        ax = axes[i]
        ax.scatter(Y_val[qoi], predictions[:, Y.columns.get_loc(qoi)], alpha=0.6, label="Previsões")
        ax.plot([Y_val[qoi].min(), Y_val[qoi].max()], [Y_val[qoi].min(), Y_val[qoi].max()], 
                color="red", linestyle="--", label="Ideal")
        ax.set_title(f"{qoi}: Y Verdadeiro vs Y Previsto")
        ax.set_xlabel("Y Verdadeiro")
        ax.set_ylabel("Y Previsto")
        ax.legend()
        ax.grid(True)

    # Desativar eixos vazios, caso o número de QoIs seja menor que o total de subplots
    for j in range(len(axes)):
        if j >= num_qois:
            axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{name}.jpg")
# Salvar os resultados em uma tabela
results_df = pd.DataFrame(results)
results_df.to_csv("resultados.csv", index=False)
print("Resultados salvos em 'resultados.csv'")
