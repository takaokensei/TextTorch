# TextTorch: Pipeline de Classificação de Texto com PyTorch

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-brightgreen.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**TextTorch** é um projeto de NLP para um seminário acadêmico, focado em criar um pipeline completo, modular e reprodutível para classificação de texto. A implementação padrão utiliza **TF-IDF** para representação de texto e um classificador feedforward simples em **PyTorch**, com a flexibilidade de trocar para **embeddings** treináveis com uma única alteração na configuração.

## Estrutura do Projeto

O projeto é organizado em uma estrutura modular para separar claramente as diferentes etapas do pipeline de machine learning.

```
TextTorch/
├─ raw/                     # Datasets brutos (ex: CSV customizado)
├─ notebooks/               # Jupyter notebooks para cada etapa do pipeline
├─ src/                     # Módulos Python com a lógica principal
├─ models/                  # Modelos treinados e arquivos de configuração
├─ artifacts/               # Artefatos gerados (vectorizer, métricas, plots)
├─ reports/                 # Relatórios e exemplos de inferência
├─ slides/                  # Apresentação do seminário
├─ requirements.txt         # Dependências do projeto
└─ README.md                # Este arquivo
```

## Como Executar

### Pré-requisitos

- Python 3.9+
- Git

### 1. Configuração do Ambiente

**a. Clone o repositório:**

```bash
git clone https://github.com/takaokensei/TextTorch.git
cd TextTorch
```

**b. Crie um ambiente virtual (recomendado):**

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\\Scripts\\activate
```

**c. Instale as dependências:**

```bash
pip install -r requirements.txt
```

### 2. Execução no Google Colab (Recomendado)

A maneira mais fácil de executar o projeto é através do Google Colab, que fornece um ambiente com GPU gratuita.

1.  **Abra o Google Colab:** [https://colab.research.google.com/](https://colab.research.google.com/)
2.  **Clone o repositório:** Em uma célula de código, execute:
    ```python
    !git clone https://github.com/takaokensei/TextTorch.git
    %cd TextTorch
    !pip install -r requirements.txt
    ```
3.  **Execute os Notebooks:** Abra os notebooks da pasta `notebooks/` em ordem sequencial (`01` a `06`). Cada notebook contém instruções detalhadas em seu cabeçalho.

### 3. Execução Local

1.  **Siga os passos de configuração do ambiente** acima.
2.  **(Opcional) Coloque o dataset customizado:** Se for usar o `Base_dados_textos_6_classes.csv`, coloque-o na pasta `raw/`.
3.  **Inicie o Jupyter:**
    ```bash
    jupyter notebook
    ```
4.  **Execute os Notebooks:** Navegue até a pasta `notebooks/` e execute os notebooks em ordem (`01` a `06`).

## Como Trocar de TF-IDF para Embeddings

O pipeline foi projetado para ser flexível. Para mudar da representação TF-IDF para embeddings treináveis:

1.  **Altere a Configuração:**
    - Abra o arquivo `models/config.yaml`.
    - Mude a linha `representation: tfidf` para `representation: embedding`.

2.  **Habilite o Código de Embedding:**
    - Em `src/representation.py`, descomente a classe `EmbeddingRepresentation`.
    - Em `src/model.py`, descomente a classe `EmbeddingClassifier` e ajuste a lógica na função `create_model`.

3.  **Reexecute os Notebooks:**
    - Execute novamente o notebook `02_representation.ipynb` para criar o vocabulário e os tensores de sequência.
    - Continue a execução a partir do notebook `03_model_definition.ipynb`.

## Artefatos Gerados

Após a execução completa dos notebooks, os seguintes artefatos serão gerados na pasta `artifacts/`:

-   `processed_dataset.pkl`: Dicionário com os dados limpos e divididos.
-   `vectorizer.joblib`: Objeto `TfidfVectorizer` treinado.
-   `tensors_tfidf.pt`: Tensores PyTorch para os dados TF-IDF.
-   `metrics.json`: Métricas de desempenho do modelo no conjunto de teste.
-   `plots/`: Gráficos como a matriz de confusão, curvas de treinamento e desempenho por classe.
