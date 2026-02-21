import pandas as pd
import os
import joblib

# 1. Pega o diretório absoluto de onde ESTE script (.py) está salvo (a pasta 'src')
DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# 2. Mapeamento das pastas relativas ao diretório do script
PATH_DADOS = os.path.abspath(os.path.join(DIRETORIO_SCRIPT, '..', 'data'))
PATH_MODELOS = os.path.abspath(os.path.join(DIRETORIO_SCRIPT, '..', 'models'))
PATH_SUBMISSION = os.path.abspath(os.path.join(DIRETORIO_SCRIPT, '..', 'submission'))

# 3. Cria as pastas caso não existam (opcional, mas seguro)
os.makedirs(PATH_DADOS, exist_ok=True)
os.makedirs(PATH_MODELOS, exist_ok=True)
os.makedirs(PATH_SUBMISSION, exist_ok=True)

print(f"Buscando dados em: {PATH_DADOS}")
print(f"Buscando modelos em: {PATH_MODELOS}")
print(f"Salvando submissões em: {PATH_SUBMISSION}\n")

print("Carregando o modelo treinado (best_model.pkl)...")
print("⚠️ NOTA: Como o modelo é uma Rede Neural, estamos carregando os pesos exatos já salvos.")
print("Isso garante total reprodutibilidade, evitando qualquer variação de resultado que um novo treinamento poderia causar.\n")

# Carrega o modelo estático já treinado
best_model = joblib.load(os.path.join(PATH_MODELOS, 'best_model_fraud.pkl'))
features_used = joblib.load(os.path.join(PATH_MODELOS, 'features_used.pkl'))
test_ids = joblib.load(os.path.join(PATH_DADOS, 'test_ids.pkl'))

print("Carregando dados de teste finais...")
X_test_final = pd.read_csv(os.path.join(PATH_DADOS, 'X_test_final.csv'))

# Garantir alinhamento estrito das colunas para evitar desalinhamento de pesos na inferência da Rede Neural
X_test_submission = X_test_final[features_used]

print("Gerando probabilidades de fraude (Classe 1)...")
# Pegamos apenas a coluna 1 que representa a probabilidade de ser fraude
submission_probs = best_model.predict_proba(X_test_submission)[:, 1]

print("Criando arquivo de submissão...")
submission = pd.DataFrame({
    'id': test_ids,
    'target': submission_probs
})

submission_file = os.path.join(PATH_SUBMISSION, 'submission_NN.csv')
submission.to_csv(submission_file, index=False)

print(f"✅ Arquivo salvo com sucesso em: {submission_file}")
print(f"   - Total de linhas: {len(submission)}")
print(f"   - Colunas geradas: {submission.columns.tolist()}")
print("\nPrimeiras linhas da submissão:")
print(submission.head())