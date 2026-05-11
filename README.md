# Categorizador de E-mails Fictícios — Projeto Aplicado

Projeto de **classificação automática de e-mails** em categorias (RH, Financeiro, Suporte e Comercial) utilizando **Naive Bayes** com vetorização **TF-IDF**, rastreamento de experimentos com **MLflow** e visualizações com **Matplotlib**.

---

## Índice

- [Visão Geral](#visão-geral)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pipeline do Projeto](#pipeline-do-projeto)
- [Como Executar](#como-executar)
- [Sprints](#sprints)
  - [Sprint 1 — Exploração e Análise dos Dados](#sprint-1--exploração-e-análise-dos-dados)
  - [Sprint 2 — Treino do Modelo](#sprint-2--treino-do-modelo)
  - [Sprint 3 — Predição de Novos E-mails](#sprint-3--predição-de-novos-e-mails)

---

## Visão Geral

O objetivo deste projeto é construir um classificador capaz de categorizar e-mails fictícios em quatro categorias:

| Categoria    | Descrição                          |
|-------------|-------------------------------------|
| **RH**       | Assuntos de recursos humanos       |
| **Financeiro** | Assuntos financeiros              |
| **Suporte**  | Solicitações de suporte técnico    |
| **Comercial** | Assuntos comerciais e vendas      |

O fluxo completo abrange desde a carga e exploração dos dados, passando pelo treinamento e avaliação do modelo, até a predição em novos dados — com rastreamento de métricas e artefatos pelo MLflow.

---

## Tecnologias Utilizadas

| Tecnologia | Uso |
|---|---|
| **Python 3.11** | Linguagem principal |
| **Pandas / NumPy** | Manipulação e análise de dados |
| **Scikit-learn 1.8** | Modelo Naive Bayes (`MultinomialNB`) e TF-IDF |
| **MLflow 3.9** | Rastreamento de experimentos, métricas e artefatos |
| **Matplotlib** | Visualização de dados (gráficos de barras e pizza) |
| **Joblib** | Serialização do vetorizador TF-IDF |
| **Jupyter Notebook** | Ambiente de desenvolvimento interativo |

---

## Estrutura do Projeto

```
PA_categorizador_emails-ficticios/
├── emails_gold.ipynb        # Notebook principal com todo o pipeline
├── emails.csv               # Dataset de e-mails fictícios (entrada)
├── tfidf_vectorizer.pkl     # Vetorizador TF-IDF serializado
├── mlflow.db                # Banco de dados local do MLflow
├── mlruns/                  # Artefatos e registros do MLflow
│   └── 1/
│       ├── <run_id>/
│       │   └── artifacts/
│       └── models/
│           └── <model_id>/
│               └── artifacts/
│                   ├── MLmodel
│                   ├── conda.yaml
│                   ├── python_env.yaml
│                   └── requirements.txt
└── README.md
```

---

## Pipeline do Projeto

```
emails.csv
    │
    ▼
┌──────────────────────────┐
│ 1. Carga e Exploração    │  ← Pandas, NumPy
│    dos Dados             │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 2. Atribuição de         │  ← Categorias: RH, Financeiro,
│    Categorias            │    Suporte, Comercial
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 3. Vetorização TF-IDF    │  ← max_features=5000
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 4. Treino Naive Bayes    │  ← 70% treino / 30% teste
│    + Logging MLflow      │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 5. Predição em Novos     │  ← Modelo + TF-IDF carregados
│    E-mails               │    do MLflow
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│ 6. Visualização dos      │  ← Gráficos de barras e pizza
│    Resultados            │
└──────────────────────────┘
```

---

## Como Executar

### Pré-requisitos

- Python 3.11+
- Jupyter Notebook ou VS Code com extensão Jupyter

### Instalação das dependências

```bash
pip install pandas numpy scikit-learn mlflow matplotlib joblib
```

### Execução

1. Certifique-se de que o arquivo `emails.csv` está na raiz do projeto.
2. Abra o notebook `emails_gold.ipynb`.
3. Execute as células sequencialmente.

### Visualizar experimentos no MLflow (opcional)

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acesse `http://localhost:5000` no navegador.

---

## Sprints

### Sprint 1 — Exploração e Análise dos Dados

**Células 1–6 do notebook**

- Importação das bibliotecas (`pandas`, `numpy`)
- Leitura do CSV (`emails.csv`)
- Criação da coluna `categoria` com valores aleatórios entre as 4 classes
- Análise exploratória: `shape`, `dtypes`, `info()`, `head()`
- Gráfico de barras com a distribuição de e-mails por categoria

### Sprint 2 — Treino do Modelo

**Células 7–11 do notebook**

- Divisão dos dados em **70% treino / 30% teste** (`train_test_split`, `random_state=42`)
- Vetorização dos textos com **TF-IDF** (`max_features=5000`)
- Treinamento do modelo **Multinomial Naive Bayes**
- Registro no **MLflow**:
  - **Parâmetros**: modelo, vetorização, max_features, test_size, random_state
  - **Métricas**: acurácia
  - **Artefatos**: modelo treinado + vetorizador TF-IDF (`tfidf_vectorizer.pkl`)
- Listagem dos runs do experimento com métricas

### Sprint 3 — Predição de Novos E-mails

**Células 12–15 do notebook**

- Carregamento de novos e-mails (simulados a partir do mesmo CSV, sem categoria)
- Recuperação do modelo e do vetorizador TF-IDF do **último run do MLflow**
- Predição das categorias com o modelo carregado
- Visualização dos resultados:
  - **Gráfico de barras**: distribuição das categorias preditas
  - **Gráfico de pizza**: proporção percentual por categoria

---

## Parâmetros do Modelo

| Parâmetro | Valor |
|---|---|
| Algoritmo | Multinomial Naive Bayes |
| Vetorização | TF-IDF |
| max_features | 5000 |
| Divisão treino/teste | 70% / 30% |
| random_state | 42 |

---

## Licença

Este projeto é de uso acadêmico/educacional.
