# Desafio Técnico - Ligia (Liga de IA da UFPE)

## Trilha: Machine Learning - Detecção de Fraudes

**Autor:** [Seu Nome Completo]
**Competição Kaggle:** [Nome ou Link da Competição]

---

## 📌 Visão Geral do Projeto

Este repositório contém a solução desenvolvida para a etapa técnica do processo seletivo da Ligia. O objetivo é desenvolver um modelo de Inteligência Artificial capaz de detectar transações financeiras fraudulentas.

O problema é caracterizado como uma tarefa de **classificação binária em dados desbalanceados**, onde a métrica principal de avaliação é a **ROC-AUC**.

### 🎯 Objetivos

1. Realizar Análise Exploratória de Dados (EDA) para identificar padrões de fraude.
2. Implementar estratégias para tratamento de classes desbalanceadas.
3. Treinar e validar modelos de Machine Learning (foco em Gradient Boosting).
4. Garantir a interpretabilidade do modelo (XAI) para justificar as decisões.
5. Gerar submissão formatada para o Kaggle.

---

## 📂 Estrutura do Repositório

A organização do código segue uma lógica de separação entre exploração, processamento e modelagem:

```text
├── data/                           # (Ignorado no Git) Pasta para datasets raw/processed
│
├── notebooks/                      # Jupyter Notebooks para análise e experimentos
│   ├── 01_eda_analise.ipynb        # Análise exploratória e visualizações
│   └── 02_modelagem_testes.ipynb   # Testes de algoritmos e validação cruzada
│
├── src/                            # Scripts Python para execução reprodutível
│   ├── preprocessing.py            # Pipelines de tratamento de dados
│   ├── train.py                    # Script principal de treinamento
│   └── inference.py                # Script para gerar o arquivo de submissão
│
├── models/                         # Artefatos serializados (modelos salvos)
│   ├── model.joblib                # Modelo final treinado
│   └── scaler.pkl                  # Scaler ajustado (se aplicável)
│
├── submission/                     # Arquivos de saída
│   └── submission.csv              # Arquivo pronto para o Kaggle
│
├── requirements.txt                # Lista de dependências do projeto
│
└── README.md                       # Documentação do projeto
```

## 🚀 Como Executar o Projeto

Para garantir a reprodutibilidade da solução, siga os passos abaixo:

### 1. Instalação das Dependências

Recomenda-se a criação de um ambiente virtual (venv ou conda).

```bash
# Clone o repositório
git clone [Link do Seu Repositório]
cd [Nome da Pasta]

# Instale os pacotes necessários
pip install -r requirements.txt

### 2. Reproduzir o Treinamento
Para treinar o modelo do zero e salvar os artefatos na pasta `models/`:
```

```bash
python src/train.py

### 3. Gerar Submissão (Inferência)
Para gerar o arquivo `.csv` com as probabilidades para o Kaggle:
```

```bash
python src/inference.py
```

## 🧠 Abordagem Técnica e Metodologia

### Pré-processamento

* **Limpeza:** Tratamento de valores nulos utilizando [ex: inputação pela mediana].
* **Feature Engineering:** Criação de novas variáveis baseadas em [ex: agregação de tempo ou valor].
* **Normalização:** Aplicação de [ex: StandardScaler ou MinMaxScaler] (se aplicável).

### Estratégia de Desbalanceamento

Dada a baixa prevalência de fraudes, foi utilizada a técnica [Escolha uma: SMOTE / Class Weights / Undersampling] para equilibrar a importância das classes durante o treinamento.

### Modelagem

* **Baseline:** Foi utilizada uma Regressão Logística simples como linha de base.
* **Modelo Final:** O algoritmo escolhido foi o **[ex: XGBoost / LightGBM]**.
* **Validação:** Stratified K-Fold Cross-Validation (5 dobras) para garantir robustez nas métricas.

### Interpretabilidade (XAI)

Para cumprir o requisito de explicabilidade "White Box", foi utilizada a biblioteca **SHAP (SHapley Additive exPlanations)**. As análises de importância das features podem ser visualizadas no notebook `notebooks/02_modelagem_testes.ipynb`.

## 📊 Resultados Preliminares

| Modelo | ROC-AUC (Validação) |
| :--- | :--- |
| Baseline (Regressão Logística) | 0.XX |
| **Modelo Proposto ([Nome])** | **0.XX** |

---

## 🛠 Tecnologias Utilizadas

* Python 3.8+
* Pandas & NumPy
* Scikit-Learn
* [XGBoost / LightGBM / CatBoost]
* SHAP (Interpretabilidade)
* Matplotlib & Seaborn