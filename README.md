<div align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=2c4c7c&height=120&section=header"/>
  
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.herokuapp.com/?lines=TextTorch+🔥;Modular+NLP+Pipeline;TF-IDF+%26+Embedding+Support;Academic+Reproducibility&font=Fira+Code&center=true&width=500&height=50&color=4A6FA5&vCenter=true&pause=1000&size=24" />
  </a>
  
  <br/>
  
  <samp>Pipeline completo, modular e reprodutível para classificação de texto acadêmica.</samp>
  
  <br/><br/>
  
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  </a>
  <a href="https://github.com/takaokensei/TextTorch/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge"/>
  </a>
</div>

<br/>

## `> about_project`

<p align="justify">
  <strong>TextTorch</strong> é um framework de NLP desenvolvido para seminários acadêmicos, focado na clareza do fluxo de dados. A implementação padrão utiliza <strong>TF-IDF</strong> para representação esparsa e um classificador feedforward em <strong>PyTorch</strong>, mas oferece flexibilidade total para alternar para <strong>embeddings treináveis</strong> (densos) através de um arquivo de configuração centralizado.
</p>

<br/>

## `> tech_stack`

<table align="center">
  <tr>
    <td align="center" width="33%">
      <strong>🔥 Core</strong>
      <br/><br/>
      <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
      <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/>
    </td>
    <td align="center" width="33%">
      <strong>📊 Data Processing</strong>
      <br/><br/>
      <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/>
      <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/>
    </td>
    <td align="center" width="33%">
      <strong>⚙️ Environment</strong>
      <br/><br/>
      <img src="https://img.shields.io/badge/Google_Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white"/>
      <img src="https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white"/>
    </td>
  </tr>
</table>

<br/>

## `> architecture`

O projeto segue uma estrutura modular rígida para garantir a separação de responsabilidades no pipeline de ML.

```bash
TextTorch/
├── 📂 raw/            # Datasets brutos (ex: CSV customizado)
├── 📓 notebooks/      # Jupyter notebooks sequenciais (01-06)
├── 📦 src/            # Lógica principal (Data Loading, Model, Train)
├── 🧠 models/         # Pesos salvos e config.yaml
├── 📊 artifacts/      # Vectorizers, plots e métricas geradas
├── 📑 reports/        # Relatórios de inferência
└── 📄 requirements.txt
```

<br/>

## `> quick_start`

### ⚡ Opção 1: Google Colab (Recomendado)

Ambiente com GPU gratuita e configuração zero.

1. Acesse o [Google Colab](https://colab.research.google.com/)
2. Clone e instale em uma célula:

```python
!git clone https://github.com/takaokensei/TextTorch.git
%cd TextTorch
!pip install -r requirements.txt
```

3. Execute os notebooks na pasta `notebooks/` sequencialmente (01 a 06)

### 🛠️ Opção 2: Execução Local

**Pré-requisitos:** Python 3.9+ e Git

```bash
# 1. Clone o repositório
git clone https://github.com/takaokensei/TextTorch.git
cd TextTorch

# 2. Crie o ambiente virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instale dependências e inicie
pip install -r requirements.txt
jupyter notebook
```

<br/>

## `> advanced_features`

### 🔄 Alternância TF-IDF ↔ Embeddings

O TextTorch permite mudar a arquitetura do modelo editando apenas a configuração, sem reescrever o código de treino.

**1. Edite a Config:**

No arquivo `models/config.yaml`, altere:

```yaml
representation: embedding  # (padrão: tfidf)
```

**2. Habilite os Módulos:**

- `src/representation.py`: Descomente `EmbeddingRepresentation`
- `src/model.py`: Descomente `EmbeddingClassifier`

**3. Re-execute:**

Rode novamente `02_representation.ipynb` (para gerar vocabulário) e `03_model_definition.ipynb`

<br/>

## `> output_artifacts`

Após a execução do pipeline, verifique a pasta `artifacts/`:

| Artefato | Descrição |
|----------|-----------|
| `processed_dataset.pkl` | Dados limpos e particionados |
| `vectorizer.joblib` | Modelo TfidfVectorizer treinado |
| `tensors_tfidf.pt` | Tensores PyTorch prontos para GPU |
| `metrics.json` | Acurácia, F1-Score e Recall finais |
| `plots/` | Matriz de confusão e curvas de aprendizado |

<br/>

---

<div align="center">
  <samp>
    <strong>📚 Desenvolvido para seminários acadêmicos @ UFRN</strong>
    <br/>
    Modular • Reprodutível • Didático
  </samp>
</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=2c4c7c&height=100&section=footer"/>
