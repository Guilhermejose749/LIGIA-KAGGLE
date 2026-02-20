import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Pega o diretório absoluto de onde ESTE script (.py) está salvo (a pasta 'src')
DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Volta um nível (..) para sair de 'src' e entra na pasta 'data'
PATH_DADOS = os.path.abspath(os.path.join(DIRETORIO_SCRIPT, '..', 'data'))

# Cria a pasta caso não exista (segurança)
os.makedirs(PATH_DADOS, exist_ok=True)

print(f"📂 Diretório de dados configurado para: {PATH_DADOS}\n")


def preprocess_data(df):
    """Aplica as transformações de engenharia de features."""
    df = df.copy()
    
    # Criar features cíclicas para o tempo
    if 'Time' in df.columns:
        df['Hour'] = df['Time'].apply(lambda x: np.floor(x / 3600) % 24)
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df.drop(columns=['Time'], inplace=True)
    
    return df

print("Carregando os dados iniciais...")
TARGET_COL = 'Class' 

df_train_raw = pd.read_csv(os.path.join(PATH_DADOS, 'train.csv'))
df_test_raw = pd.read_csv(os.path.join(PATH_DADOS, 'test.csv'))

print("Aplicando Feature Engineering (transformações cíclicas de tempo)...")
df_train_processed = preprocess_data(df_train_raw)
df_test_processed = preprocess_data(df_test_raw)

# Separar os IDs do conjunto de teste para a submissão final
test_ids = df_test_processed['id'].copy()

# Remover a coluna 'id' de ambos os datasets
df_train_processed.drop(columns=['id'], inplace=True, errors='ignore')
df_test_processed.drop(columns=['id'], inplace=True, errors='ignore')

# Salvar os dataframes processados integrais
df_train_processed.to_csv(os.path.join(PATH_DADOS, 'df_train_processed.csv'), index=False)
df_test_processed.to_csv(os.path.join(PATH_DADOS, 'df_test_processed.csv'), index=False)

print("Separando features e target...")
X = df_train_processed.drop(columns=[TARGET_COL])
y = df_train_processed[TARGET_COL]
X_test_final = df_test_processed.copy()

# Garantir que as colunas do teste estejam na mesma ordem do treino
X_test_final = X_test_final[X.columns]

print("Dividindo em treino e validação (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Aplicando RobustScaler nas features...")
scaler = RobustScaler()

# Ajustar e transformar o treino; apenas transformar validação e teste
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
X_test_final_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns)

print("Salvando artefatos e bases de dados escalonadas...")
joblib.dump(scaler, os.path.join(PATH_DADOS, 'robust_scaler.pkl'))
joblib.dump(test_ids, os.path.join(PATH_DADOS, 'test_ids.pkl')) # Salvando IDs para uso no script 3

X_train_scaled.to_csv(os.path.join(PATH_DADOS, 'X_train.csv'), index=False)
X_val_scaled.to_csv(os.path.join(PATH_DADOS, 'X_val.csv'), index=False)
y_train.to_csv(os.path.join(PATH_DADOS, 'y_train.csv'), index=False)
y_val.to_csv(os.path.join(PATH_DADOS, 'y_val.csv'), index=False)
X_test_final_scaled.to_csv(os.path.join(PATH_DADOS, 'X_test_final.csv'), index=False)

print("\n✅ Etapa 1 concluída com sucesso! Dados preparados e salvos em '../data'.")