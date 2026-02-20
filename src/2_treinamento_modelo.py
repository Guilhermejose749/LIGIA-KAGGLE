import pandas as pd
import os
import joblib
from sklearn.neural_network import MLPClassifier

# 1. Pega o diretório absoluto de onde ESTE script (.py) está salvo (a pasta 'src')
DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# 2. Mapeamento das pastas relativas ao diretório do script
PATH_DADOS = os.path.abspath(os.path.join(DIRETORIO_SCRIPT, '..', 'data'))
PATH_MODELOS = os.path.abspath(os.path.join(DIRETORIO_SCRIPT, '..', 'models'))

# 3. Cria as pastas caso não existam (opcional, mas seguro)
os.makedirs(PATH_DADOS, exist_ok=True)
os.makedirs(PATH_MODELOS, exist_ok=True)

print(f"Buscando dados em: {PATH_DADOS}")
print(f"Salvando modelos em: {PATH_MODELOS}\n")

print("Carregando bases de treinamento...")
X_train = pd.read_csv(os.path.join(PATH_DADOS, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(PATH_DADOS, 'y_train.csv')).squeeze() # squeeze para converter em Series

# Salvando a lista de features usadas para validação futura
features_usadas = X_train.columns.tolist()
joblib.dump(features_usadas, os.path.join(PATH_MODELOS, 'features_used.pkl'))

# ==============================================================================
# AVISO SOBRE O PROPÓSITO DESTE SCRIPT
# ==============================================================================
print("\n" + "="*75)
print("⚠️ AVISO IMPORTANTE SOBRE ESTE SCRIPT DE TREINAMENTO")
print("="*75)
print("Este script tem o propósito de DEMONSTRAR como o modelo final foi criado")
print("e consolidar os hiperparâmetros ótimos encontrados durante o tuning.")
print("\nComo modelos de Redes Neurais possuem inicialização aleatória de pesos")
print("e otimização estocástica, treiná-lo novamente (mesmo com random_state fixo)")
print("pode gerar resultados ligeiramente diferentes devido a variações de hardware.")
print("\nPor isso, este script salvará o resultado como 'best_neural_net_model.pkl'.")
print("No entanto, o script de submissão utilizará o arquivo 'best_model.pkl'")
print("original, garantindo a reprodutibilidade exata do nosso melhor score.")
print("="*75 + "\n")

print("Configurando a Rede Neural (MLPClassifier)...")
# Parâmetros definidos pelo seu tuning
mlp = MLPClassifier(
    activation='tanh',
    alpha=0.00011727009450102256,
    batch_size='auto',
    beta_1=0.9,
    beta_2=0.999,
    early_stopping=True,
    epsilon=1e-08,
    hidden_layer_sizes=(100, 50),
    learning_rate='constant',
    learning_rate_init=0.006847920095574782,
    max_fun=15000,
    max_iter=500,
    momentum=0.9,
    n_iter_no_change=10,
    nesterovs_momentum=True,
    power_t=0.5,
    shuffle=True,
    solver='adam',
    tol=0.0001,
    validation_fraction=0.1,
    warm_start=False,
    random_state=42 # Fixado para tentar manter a maior reprodutibilidade possível
)

print("Iniciando o treinamento do modelo demonstrativo (isso pode levar alguns minutos)...")
mlp.fit(X_train, y_train)

print("Salvando o modelo demonstrativo...")
# Salvando com o nome indicativo para não sobrescrever o 'best_model.pkl' original
joblib.dump(mlp, os.path.join(PATH_MODELOS, 'best_neural_net_model.pkl'))

print("\n✅ Etapa 2 concluída com sucesso! Modelo demonstrativo salvo como 'best_neural_net_model.pkl'.")