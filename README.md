#  Fraud Detection - Ligia Machine Learning Challenge 2026

Projeto de Machine Learning para detecção de fraudes em transações financeiras, desenvolvido para o desafio técnico da **Ligia (Liga de IA da UFPE)**.

O objetivo é classificar transações como fraudulentas ou legítimas em um cenário altamente desbalanceado.

---

#  Dataset

Este projeto utiliza o dataset oficial da competição Ligia:

```bash
kaggle competitions download -c ligia-machine-learning
```

O dataset é composto por dois arquivos principais:

- `train.csv` → Contém as features + variável target (rótulo de fraude)
- `test.csv` → Contém apenas as features (sem target), utilizado exclusivamente para geração da submissão

---

## Estratégia de Uso dos Dados

- O modelo foi **treinado e validado exclusivamente utilizando o `train.csv`**.
- A validação foi realizada internamente via **Cross-Validation (Stratified K-Fold)**.
- O arquivo `test.csv` **não foi utilizado em nenhum momento do treinamento**.
- O `test.csv` foi usado apenas para:
  - Gerar as probabilidades de fraude
  - Criar o arquivo `submission.csv`
  - Enviar a submissão para avaliação no Kaggle

Essa separação garante:

- Ausência de vazamento de dados (data leakage)
- Avaliação justa no leaderboard
- Reprodutibilidade do experimento

---

## Características do Problema

O desafio é caracterizado como:

- Classificação binária (Fraude vs Não Fraude)
- Dados altamente desbalanceados
- Métrica principal de avaliação: **ROC-AUC**
- Foco em performance da classe minoritária (fraudes)

---

#  Features do DataSet

O dataset contém variáveis anônimas (V1, V2, ..., V28), além de:

- `Time` → segundos decorridos desde a primeira transação
- `Amount` → valor da transação
- `Class` → variável alvo (0 = normal, 1 = fraude)

---

##  Feature Engineering

Além das variáveis originais, foram criadas novas features baseadas no comportamento temporal:

###  1. Hour

Conversão da variável `Time` para hora do dia:

```
Hour = (Time / 3600) % 24
```

Isso permite capturar padrões comportamentais ao longo do dia.

---

###  2. Transformação Cíclica da Hora

Como horas possuem natureza circular (23h → 0h), foram criadas duas transformações trigonométricas:

```
Hour_sin = sin(2π * Hour / 24)
Hour_cos = cos(2π * Hour / 24)
```

Essas variáveis ajudam o modelo a aprender padrões temporais de forma contínua e não linear.

---

#  Modelos Testados

Foram avaliados 6 modelos diferentes:

1. Regressão Logística
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost
6. Neural Network (MLPClassifier)

Todos foram avaliados com validação cruzada estratificada utilizando **ROC-AUC** como métrica principal.

---

#  Modelo Final

O modelo que apresentou **maior ROC-AUC no conjunto de teste público do desafio** foi:

##  Neural Network (MLPClassifier)

---

#  Estratégia de Treinamento

O modelo foi otimizado utilizando:

- RandomizedSearchCV
- Validação cruzada (Stratified K-Fold)
- Early Stopping
- Métrica de otimização: **ROC-AUC**

---

##  Intervalos de Busca Avaliados

Durante o Random Search, os seguintes intervalos foram explorados:

- hidden_layer_sizes:
  - (50,)
  - (100,)
  - (100, 50)
  - (64, 32)

- activation:
  - relu
  - tanh

- alpha:
  - intervalo log-uniforme entre 1e-4 e 1e-1

- learning_rate_init:
  - intervalo log-uniforme entre 1e-4 e 1e-1

- solver:
  - adam

---

#  Hiperparâmetros Finais do Modelo

hidden_layer_sizes : (100, 50)  
activation         : tanh  
solver             : adam  
alpha              : 0.00011727009450102256  
learning_rate_init : 0.006847920095574782  

---

# Performance do Modelo

 MÉTRICAS DA CLASSE MINORITÁRIA (Fraude):

• Recall (Detecção)   : 75.95%  
• Precisão (Acerto)   : 83.33%  
• F1-Score (Balanço)  : 0.7947  
• ROC-AUC             : 0.9762  


##  Interpretação

- O modelo captura aproximadamente 76% das fraudes reais.
- Mantém boa precisão, reduzindo bloqueios indevidos.
- Apresenta excelente capacidade de separação entre classes (ROC-AUC 0.9762).
- Superou todos os demais modelos testados no leaderboard público.

---

#  Setup do Ambiente

**Pré-requisito:** Este projeto requer o **Python 3.11**. Certifique-se de ter a versão correta instalada antes de prosseguir.

Este projeto pode ser executado de três formas:

- Com Conda (ambiente isolado)
- Com venv
- Sem ambiente virtual (instalação global)

---

# Opção 1 — Usando Conda (Recomendado)

```bash
conda env create -f environment.yml
conda activate ligia_fraud
```

Ou manualmente:

```bash
conda create -n ligia_fraud python=3.11 -y
conda activate ligia_fraud
pip install -r requirements.txt
```

---

# Opção 2 — Usando venv

```bash
# Linux/Mac
python3.11 -m venv venv
source venv/bin/activate  

# Windows
py -3.11 -m venv venv
venv\Scripts\activate

# Com o ambiente ativado, instale as dependências:
pip install -r requirements.txt
```

---

# Opção 3 — Sem Ambiente Virtual

⚠️ Não recomendado para produção.

```bash
pip install -r requirements.txt
```

---

#  Verificação da Instalação

```bash
python -c "import numpy, pandas, sklearn; print('Ambiente configurado!')"
```

---

#  Fluxo de Execução

O projeto pode ser executado de **duas formas distintas**, dependendo do objetivo:

- **Modo 1 — Execução via Scripts (`src/`)** → Pipeline reprodutível para gerar dados processados, treinar o modelo final e criar a submissão.
- **Modo 2 — Execução via Notebooks** → Análise exploratória completa, comparação entre modelos e geração de gráficos.

---

## Método 1 — Execução via Scripts (Pipeline Reprodutível)

Recomendado para quem deseja:

- Reproduzir o modelo final
- Gerar arquivos processados
- Treinar a Neural Network otimizada
- Criar o `submission.csv`

### Preparação dos Dados

Certifique-se de que `train.csv` e `test.csv` estão na pasta `data/`.

---

### Pré-processamento

```bash
python src/1_preparar_dados.py
```

Este script:

- Realiza limpeza
- Aplica feature engineering
- Cria `Hour`, `Hour_sin` e `Hour_cos`
- Gera arquivos processados em `data/`

---

### Treinamento do Modelo

```bash
python src/2_treinamento_modelo.py
```

Este script:

- Executa o RandomizedSearchCV
- Seleciona o melhor modelo
- Salva o modelo final em `models/`

---

### Inferência e Geração da Submissão

```bash
python src/3_inferencia.py
```

Este script:

- Carrega o modelo salvo
- Gera probabilidades para o `test.csv`
- Cria o arquivo final:

```
submission/submission.csv
```

### Pipeline completa

```bash
python src/1_preparar_dados.py

python src/2_treinamento_modelo.py

python src/3_inferencia.py
```

---

## Método 2 — Execução via Notebooks (Análise Completa)

Recomendado para:

- Exploração detalhada dos dados
- Visualização de gráficos
- Comparação entre os 6 modelos
- Análise das métricas
- Interpretação com SHAP

### Iniciar Jupyter

```bash
jupyter notebook
```

ou

```bash
jupyter lab
```

### Ordem de execução

1. `1.0_analise_exploratoria.ipynb`
2. `1.1_feature_engineering.ipynb`
3. `2.0_treinamento_modelos.ipynb`
4. `2.1_criar_arquivos_submission.ipynb`
5. `3.0_analise_modelo_escolhido.ipynb`

---

#  Organização do Projeto

```
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── data
│   ├── test.csv
│   └── train.csv
├── environment.yml
├── models
│   ├── best_model_fraud.pkl
│   ├── features_used.pkl
│   └── new_models
│       ├── modelo_catboost.pkl
│       ├── modelo_lightgbm.pkl
│       ├── modelo_neural_net.pkl
│       ├── modelo_random_forest.pkl
│       ├── modelo_regressao_logistica.pkl
│       └── modelo_xgboost.pkl
├── notebooks
│   ├── 1.0_analise_exploratoria.ipynb
│   ├── 1.1_feature_engineering.ipynb
│   ├── 2.0_treinamento_modelos.ipynb
│   ├── 2.1_criar_arquivos_submission.ipynb
│   └── 3.0_analise_modelo_escolhido.ipynb
├── requirements.txt
├── src
│   ├── 1_preparar_dados.py
│   ├── 2_treinamento_modelo.py
│   └── 3_inferencia.py
└── submission
    └── submission.csv
```

---

#  Organização do Projeto após execução dos notebooks

```
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── data
│   ├── X_test_final.csv
│   ├── X_train.csv
│   ├── X_val.csv
│   ├── df_test_processed.csv
│   ├── df_train_processed.csv
│   ├── robust_scaler.pkl
│   ├── test.csv
│   ├── train.csv
│   ├── y_train.csv
│   └── y_val.csv
├── environment.yml
├── models
│   ├── best_model_fraud.pkl
│   ├── features_used.pkl
│   └── new_models
│       ├── modelo_catboost.pkl
│       ├── modelo_lightgbm.pkl
│       ├── modelo_neural_net.pkl
│       ├── modelo_random_forest.pkl
│       ├── modelo_regressao_logistica.pkl
│       └── modelo_xgboost.pkl
├── notebooks
│   ├── 1.0_analise_exploratoria.ipynb
│   ├── 1.1_feature_engineering.ipynb
│   ├── 2.0_treinamento_modelos.ipynb
│   ├── 2.1_criar_arquivos_submission.ipynb
│   └── 3.0_analise_modelo_escolhido.ipynb
├── paper_artifacts
│   ├── barras_metricas_baseline.png
│   ├── business_report_Neural_net.txt
│   ├── business_report_XGBoost.txt
│   ├── confusion_matrix_Neural_net.png
│   ├── confusion_matrix_XGBoost.png
│   ├── curvas_comparativas_baseline.png
│   ├── roc_curve_Neural_net.png
│   ├── roc_curve_XGBoost.png
│   ├── shap_numerical_importance_Neural_net.csv
│   ├── shap_numerical_importance_XGBoost.csv
│   ├── shap_summary_Neural_net.png
│   └── shap_summary_XGBoost.png
├── requirements.txt
├── src
│   ├── 1_preparar_dados.py
│   ├── 2_treinamento_modelo.py
│   └── 3_inferencia.py
└── submission
    ├── others
    │   ├── submission_Cat.csv
    │   ├── submission_Light.csv
    │   └── submission_XGB.csv
    └── submission.csv
```

---

#  Conclusão

Este projeto demonstra uma abordagem comparativa robusta entre múltiplos modelos de classificação, com foco em:

- Performance em dados desbalanceados
- Otimização via Random Search
- Feature engineering temporal
- Avaliação baseada em métrica de negócio (ROC-AUC)

A Neural Network apresentou o melhor desempenho no leaderboard público do desafio, tornando-se o modelo final selecionado.