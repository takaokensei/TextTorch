"""
Streamlit app for TextTorch NLP pipeline.
Interactive interface for dataset selection, training, evaluation, and inference.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Optional

# Adiciona utils e src ao path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import (
    load_and_prepare_dataset,
    create_vectorizer,
    vectorize_texts,
    create_model_from_config,
    train_model_wrapper,
    evaluate_model_wrapper,
    predict_text,
    load_config,
    save_config,
    get_device_wrapper,
    find_misclassified
)

# Importar funções de plot do evaluate
from evaluate import plot_confusion_matrix, plot_training_curves, plot_class_accuracy

# Configuração da página
st.set_page_config(
    page_title="TextTorch Explorer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Inicializar session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_history' not in st.session_state:
    st.session_state.train_history = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'device' not in st.session_state:
    st.session_state.device = None


def load_default_config():
    """Carrega configuração padrão do arquivo."""
    try:
        return load_config('../models/config.yaml')
    except FileNotFoundError:
        # Configuração padrão se arquivo não existir
        return {
            'representation': 'tfidf',
            'model_type': 'feedforward',
            'input_dim': 'auto',
            'hidden_dim': 512,
            'dropout': 0.5,
            'n_classes': 'auto',
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10,
            'weight_decay': 0.0001,
            'embedding_dim': 300,
            'vocab_size': 'auto',
            'pretrained_embeddings': None,
            'seed': 42,
            'dataset': '20newsgroups',
            'test_size': 0.1,
            'val_size': 0.1,
            'min_df': 5,
            'max_df': 0.8,
            'ngram_range': [1, 2],
            'max_features': 10000,
            'early_stopping': True,
            'patience': 3
        }


# Carregar configuração inicial
if st.session_state.config is None:
    st.session_state.config = load_default_config()

# ============================================
# SIDEBAR - Configurações
# ============================================
st.sidebar.title("⚙️ Configurações")

# Dataset
st.sidebar.subheader("📊 Dataset")
dataset_choice = st.sidebar.selectbox(
    "Selecione o dataset:",
    ['20newsgroups', 'custom_csv'],
    index=0 if st.session_state.config.get('dataset') == '20newsgroups' else 1
)

csv_text_col = 'text'
csv_label_col = 'label'

if dataset_choice == 'custom_csv':
    st.sidebar.info("💡 Para usar CSV customizado, certifique-se de que o arquivo está em `raw/Base_dados_textos_6_classes.csv`")
    csv_text_col = st.sidebar.text_input(
        "Coluna de texto:",
        value=st.session_state.config.get('csv_text_col', 'Texto Expandido')
    )
    csv_label_col = st.sidebar.text_input(
        "Coluna de label:",
        value=st.session_state.config.get('csv_label_col', 'Classe')
    )

# Representação
st.sidebar.subheader("🔤 Representação")
representation = st.sidebar.selectbox(
    "Tipo de representação:",
    ['tfidf', 'embedding'],
    index=0 if st.session_state.config.get('representation') == 'tfidf' else 1
)

if representation == 'embedding':
    st.sidebar.warning("⚠️ Embeddings ainda não estão implementados. Use TF-IDF por enquanto.")

# Hiperparâmetros do Modelo
st.sidebar.subheader("🧠 Modelo")
hidden_dim = st.sidebar.slider(
    "Dimensão da camada oculta:",
    min_value=64,
    max_value=1024,
    value=st.session_state.config.get('hidden_dim', 512),
    step=64
)
dropout = st.sidebar.slider(
    "Dropout:",
    min_value=0.0,
    max_value=0.9,
    value=st.session_state.config.get('dropout', 0.5),
    step=0.1
)

# Hiperparâmetros de Treinamento
st.sidebar.subheader("🎯 Treinamento")
batch_size = st.sidebar.selectbox(
    "Batch size:",
    [16, 32, 64, 128],
    index=[16, 32, 64, 128].index(st.session_state.config.get('batch_size', 32))
)
learning_rate = st.sidebar.number_input(
    "Learning rate:",
    min_value=0.0001,
    max_value=0.01,
    value=float(st.session_state.config.get('learning_rate', 0.001)),
    step=0.0001,
    format="%.4f"
)
epochs = st.sidebar.number_input(
    "Épocas:",
    min_value=1,
    max_value=100,
    value=int(st.session_state.config.get('epochs', 10)),
    step=1
)
weight_decay = st.sidebar.number_input(
    "Weight decay:",
    min_value=0.0,
    max_value=0.001,
    value=float(st.session_state.config.get('weight_decay', 0.0001)),
    step=0.0001,
    format="%.4f"
)
early_stopping = st.sidebar.checkbox(
    "Early stopping:",
    value=st.session_state.config.get('early_stopping', True)
)
patience = st.sidebar.number_input(
    "Paciência (early stopping):",
    min_value=1,
    max_value=10,
    value=int(st.session_state.config.get('patience', 3)),
    step=1,
    disabled=not early_stopping
)

# Parâmetros TF-IDF
if representation == 'tfidf':
    st.sidebar.subheader("📝 TF-IDF")
    min_df = st.sidebar.number_input(
        "Min DF:",
        min_value=1,
        max_value=10,
        value=int(st.session_state.config.get('min_df', 5)),
        step=1
    )
    max_df = st.sidebar.slider(
        "Max DF:",
        min_value=0.1,
        max_value=1.0,
        value=float(st.session_state.config.get('max_df', 0.8)),
        step=0.1
    )
    max_features = st.sidebar.selectbox(
        "Max features:",
        [5000, 10000, 20000, 50000],
        index=[5000, 10000, 20000, 50000].index(st.session_state.config.get('max_features', 10000))
    )

# Data Split
st.sidebar.subheader("📂 Divisão dos Dados")
test_size = st.sidebar.slider(
    "Test size:",
    min_value=0.05,
    max_value=0.3,
    value=float(st.session_state.config.get('test_size', 0.1)),
    step=0.05
)
val_size = st.sidebar.slider(
    "Val size:",
    min_value=0.05,
    max_value=0.3,
    value=float(st.session_state.config.get('val_size', 0.1)),
    step=0.05
)

# Validação
if test_size + val_size >= 1.0:
    st.sidebar.error("❌ A soma de test_size e val_size deve ser menor que 1.0!")

# Seed
st.sidebar.subheader("🎲 Reprodutibilidade")
seed = st.sidebar.number_input(
    "Seed:",
    min_value=0,
    max_value=9999,
    value=int(st.session_state.config.get('seed', 42)),
    step=1
)

# Botão para salvar configuração
st.sidebar.markdown("---")
if st.sidebar.button("💾 Salvar Configuração", use_container_width=True):
    new_config = {
        'representation': representation,
        'model_type': 'feedforward' if representation == 'tfidf' else 'lstm',
        'input_dim': 'auto',
        'hidden_dim': hidden_dim,
        'dropout': dropout,
        'n_classes': 'auto',
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'embedding_dim': 300,
        'vocab_size': 'auto',
        'pretrained_embeddings': None,
        'seed': seed,
        'dataset': dataset_choice,
        'test_size': test_size,
        'val_size': val_size,
        'min_df': min_df if representation == 'tfidf' else 5,
        'max_df': max_df if representation == 'tfidf' else 0.8,
        'ngram_range': [1, 2],
        'max_features': max_features if representation == 'tfidf' else 10000,
        'early_stopping': early_stopping,
        'patience': patience
    }
    
    if dataset_choice == 'custom_csv':
        new_config['csv_text_col'] = csv_text_col
        new_config['csv_label_col'] = csv_label_col
    
    try:
        save_config(new_config, '../models/config.yaml')
        st.session_state.config = new_config
        st.sidebar.success("✅ Configuração salva com sucesso!")
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao salvar: {e}")

# Informações do dispositivo
st.sidebar.markdown("---")
st.sidebar.subheader("💻 Dispositivo")
if st.session_state.device is None:
    st.session_state.device = get_device_wrapper()
st.sidebar.info(f"Dispositivo: **{st.session_state.device.upper()}**")

# ============================================
# MAIN AREA
# ============================================
st.title("🔥 TextTorch Explorer")
st.markdown("Interface interativa para classificação de texto com PyTorch")

# ============================================
# Seção 1: Carregamento de Dados
# ============================================
st.header("📊 1. Carregamento de Dados")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🔄 Carregar Dataset", use_container_width=True):
        try:
            with st.spinner("Carregando dataset..."):
                csv_path = '../raw/Base_dados_textos_6_classes.csv' if dataset_choice == 'custom_csv' else None
                
                # Validação de parâmetros
                if test_size + val_size >= 1.0:
                    st.error("❌ A soma de test_size e val_size deve ser menor que 1.0!")
                else:
                    dataset = load_and_prepare_dataset(
                        dataset_choice=dataset_choice,
                        csv_path=csv_path if dataset_choice == 'custom_csv' else '../raw/Base_dados_textos_6_classes.csv',
                        csv_text_col=csv_text_col if dataset_choice == 'custom_csv' else 'text',
                        csv_label_col=csv_label_col if dataset_choice == 'custom_csv' else 'label',
                        test_size=test_size,
                        val_size=val_size,
                        seed=seed
                    )
                    
                    st.session_state.dataset = dataset
                    st.success("✅ Dataset carregado com sucesso!")
        except Exception as e:
            st.error(f"❌ Erro ao carregar dataset: {str(e)}")

with col2:
    if st.session_state.dataset is not None:
        st.metric("Status", "✅ Carregado")

if st.session_state.dataset is not None:
    dataset = st.session_state.dataset
    
    # Estatísticas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Treino", f"{len(dataset['X_train']):,}")
    with col2:
        st.metric("Validação", f"{len(dataset['X_val']):,}")
    with col3:
        st.metric("Teste", f"{len(dataset['X_test']):,}")
    with col4:
        st.metric("Classes", dataset['n_classes'])
    
    # Distribuição de classes
    st.subheader("Distribuição de Classes (Treino)")
    
    # Cria DataFrame para visualização
    df_train = pd.DataFrame({
        'label_idx': dataset['y_train'],
        'label_name': [dataset['idx_to_label'][idx] for idx in dataset['y_train']]
    })
    
    class_counts = df_train['label_name'].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.95, len(class_counts)))
    bars = ax.barh(range(len(class_counts)), class_counts.values, color=colors)
    
    ax.set_yticks(range(len(class_counts)))
    ax.set_yticklabels(class_counts.index)
    ax.set_xlabel('Número de Amostras', fontsize=12)
    ax.set_title('Distribuição de Classes no Conjunto de Treino', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Adiciona valores nas barras
    for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
        ax.text(count + max(class_counts.values) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{count:,}', va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================
# Seção 2: Treinamento
# ============================================
st.header("🎯 2. Treinamento")

if st.session_state.dataset is None:
    st.warning("⚠️ Carregue o dataset primeiro!")
else:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🚀 Treinar Modelo", use_container_width=True):
            try:
                dataset = st.session_state.dataset
                
                # Atualiza config com valores atuais
                current_config = st.session_state.config.copy()
                current_config.update({
                    'hidden_dim': hidden_dim,
                    'dropout': dropout,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'weight_decay': weight_decay,
                    'early_stopping': early_stopping,
                    'patience': patience,
                    'min_df': min_df if representation == 'tfidf' else 5,
                    'max_df': max_df if representation == 'tfidf' else 0.8,
                    'max_features': max_features if representation == 'tfidf' else 10000,
                    'model_save_path': f'../models/{representation}_model.pt'
                })
                
                with st.spinner("Treinando modelo..."):
                    # 1. Criar vectorizer
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Criando vectorizer...")
                    vectorizer = create_vectorizer(current_config)
                    progress_bar.progress(10)
                    
                    # 2. Vectorizar textos
                    status_text.text("Vectorizando textos de treino...")
                    X_train_tensor = vectorize_texts(vectorizer, dataset['X_train'], fit=True)
                    progress_bar.progress(30)
                    
                    status_text.text("Vectorizando textos de validação...")
                    X_val_tensor = vectorize_texts(vectorizer, dataset['X_val'], fit=False)
                    progress_bar.progress(40)
                    
                    # 3. Converter labels para tensores
                    status_text.text("Preparando labels...")
                    y_train_tensor = torch.tensor(dataset['y_train'], dtype=torch.long)
                    y_val_tensor = torch.tensor(dataset['y_val'], dtype=torch.long)
                    progress_bar.progress(50)
                    
                    # 4. Criar DataLoaders
                    status_text.text("Criando DataLoaders...")
                    from torch.utils.data import TensorDataset, DataLoader
                    
                    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    progress_bar.progress(60)
                    
                    # 5. Criar modelo
                    status_text.text("Criando modelo...")
                    input_dim = X_train_tensor.shape[1]
                    n_classes = dataset['n_classes']
                    
                    model = create_model_from_config(current_config, input_dim, n_classes)
                    progress_bar.progress(70)
                    
                    # 6. Treinar
                    status_text.text("Treinando modelo...")
                    
                    # Placeholder para métricas em tempo real
                    metrics_placeholder = st.empty()
                    
                    # Passa idx_to_label para salvar no checkpoint
                    idx_to_label = dataset['idx_to_label']
                    
                    history = train_model_wrapper(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        config=current_config,
                        idx_to_label=idx_to_label
                    )
                    
                    progress_bar.progress(100)
                    status_text.text("Treinamento concluído!")
                    
                    # Salvar vectorizer
                    vectorizer.save('../artifacts/vectorizer.joblib')
                    
                    # Atualizar session state
                    st.session_state.vectorizer = vectorizer
                    st.session_state.model = model
                    st.session_state.train_history = history
                    st.session_state.config = current_config
                    
                    st.success("✅ Modelo treinado com sucesso!")
                    
                    # Exibir métricas finais
                    st.subheader("Métricas de Treinamento")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Épocas Treinadas", history['epochs_trained'])
                        st.metric("Tempo Total", f"{history['training_time']/60:.2f} min")
                    with col2:
                        if history['train_losses']:
                            st.metric("Loss Final (Treino)", f"{history['train_losses'][-1]:.4f}")
                        if history['val_losses']:
                            st.metric("Loss Final (Val)", f"{history['val_losses'][-1]:.4f}")
                    
            except Exception as e:
                st.error(f"❌ Erro ao treinar modelo: {str(e)}")
                import traceback
                with st.expander("Detalhes do erro"):
                    st.code(traceback.format_exc())
    
    with col2:
        if st.session_state.model is not None:
            st.metric("Status", "✅ Treinado")

# ============================================
# Seção 3: Avaliação
# ============================================
st.header("📈 3. Avaliação")

if st.session_state.model is None:
    st.warning("⚠️ Treine o modelo primeiro!")
else:
    if st.button("📊 Avaliar Modelo", use_container_width=True):
        try:
            dataset = st.session_state.dataset
            model = st.session_state.model
            
            with st.spinner("Avaliando modelo..."):
                # Vectorizar textos de teste
                X_test_tensor = vectorize_texts(st.session_state.vectorizer, dataset['X_test'], fit=False)
                y_test_tensor = torch.tensor(dataset['y_test'], dtype=torch.long)
                
                # Criar DataLoader
                from torch.utils.data import TensorDataset, DataLoader
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                
                # Avaliar
                class_names = [dataset['idx_to_label'][i] for i in range(dataset['n_classes'])]
                metrics = evaluate_model_wrapper(
                    model=model,
                    dataloader=test_loader,
                    device=st.session_state.device,
                    class_names=class_names
                )
                
                st.session_state.metrics = metrics
                
                st.success("✅ Avaliação concluída!")
        
        except Exception as e:
            st.error(f"❌ Erro ao avaliar modelo: {str(e)}")
            import traceback
            with st.expander("Detalhes do erro"):
                st.code(traceback.format_exc())
    
    if st.session_state.metrics is not None:
        metrics = st.session_state.metrics
        
        # Métricas principais
        st.subheader("Métricas Principais")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("F1-Score (Macro)", f"{metrics['f1_macro']:.4f}")
        with col3:
            st.metric("Precision (Macro)", f"{metrics['precision_macro']:.4f}")
        with col4:
            st.metric("Recall (Macro)", f"{metrics['recall_macro']:.4f}")
        
        # Gráficos
        st.subheader("Visualizações")
        
        # Matriz de confusão
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Matriz de Confusão")
            cm = np.array(metrics['confusion_matrix'])
            class_names = [st.session_state.dataset['idx_to_label'][i] for i in range(len(cm))]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar_kws={'label': 'Porcentagem (%)'}, ax=ax)
            ax.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
            ax.set_ylabel('Classe Real', fontsize=12)
            ax.set_xlabel('Classe Predita', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### F1-Score por Classe")
            f1_scores = metrics['f1_per_class']
            sorted_indices = np.argsort(f1_scores)
            sorted_classes = [class_names[i] for i in sorted_indices]
            sorted_f1 = [f1_scores[i] for i in sorted_indices]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_classes)))
            bars = ax.barh(range(len(sorted_classes)), sorted_f1, color=colors)
            ax.set_yticks(range(len(sorted_classes)))
            ax.set_yticklabels(sorted_classes)
            ax.set_xlabel('F1-Score', fontsize=12)
            ax.set_title('F1-Score por Classe', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1.0)
            
            for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
                ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Curvas de treinamento
        if st.session_state.train_history is not None:
            st.markdown("#### Curvas de Treinamento")
            history = st.session_state.train_history
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            epochs = range(1, len(history['train_losses']) + 1)
            
            ax1.plot(epochs, history['train_losses'], 'b-', label='Treino', linewidth=2)
            if history['val_losses']:
                ax1.plot(epochs, history['val_losses'], 'r-', label='Validação', linewidth=2)
            ax1.set_title('Loss ao Longo das Épocas', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Época', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.plot(epochs, history['train_accs'], 'b-', label='Treino', linewidth=2)
            if history['val_accs']:
                ax2.plot(epochs, history['val_accs'], 'r-', label='Validação', linewidth=2)
            ax2.set_title('Acurácia ao Longo das Épocas', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Época', fontsize=12)
            ax2.set_ylabel('Acurácia', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Exemplos mal classificados
        st.subheader("Exemplos Mal Classificados (Top 10)")
        misclassified = find_misclassified(
            metrics=metrics,
            texts=st.session_state.dataset['X_test'],
            class_names=class_names,
            top_n=10
        )
        
        if misclassified:
            df_mis = pd.DataFrame(misclassified)
            st.table(df_mis)
        else:
            st.info("🎉 Nenhum exemplo mal classificado encontrado!")
        
        # Botões de exportação
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Salvar Métricas", use_container_width=True):
                try:
                    import json
                    os.makedirs('../artifacts', exist_ok=True)
                    metrics_to_save = {
                        'accuracy': metrics['accuracy'],
                        'precision_macro': metrics['precision_macro'],
                        'recall_macro': metrics['recall_macro'],
                        'f1_macro': metrics['f1_macro'],
                        'precision_micro': metrics['precision_micro'],
                        'recall_micro': metrics['recall_micro'],
                        'f1_micro': metrics['f1_micro'],
                        'precision_per_class': metrics['precision_per_class'],
                        'recall_per_class': metrics['recall_per_class'],
                        'f1_per_class': metrics['f1_per_class'],
                        'confusion_matrix': metrics['confusion_matrix'],
                        'classification_report': metrics['classification_report']
                    }
                    with open('../artifacts/metrics.json', 'w', encoding='utf-8') as f:
                        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
                    st.success("✅ Métricas salvas em `artifacts/metrics.json`")
                except Exception as e:
                    st.error(f"❌ Erro ao salvar: {e}")
        
        with col2:
            if st.button("📊 Salvar Gráficos", use_container_width=True):
                try:
                    os.makedirs('../artifacts/plots', exist_ok=True)
                    
                    # Salvar matriz de confusão
                    plot_confusion_matrix(
                        np.array(metrics['confusion_matrix']),
                        class_names,
                        '../artifacts/plots/confusion_matrix.png'
                    )
                    
                    # Salvar curvas de treinamento
                    if st.session_state.train_history:
                        plot_training_curves(
                            st.session_state.train_history,
                            '../artifacts/plots/training_curves.png'
                        )
                    
                    # Salvar F1 por classe
                    plot_class_accuracy(
                        metrics,
                        class_names,
                        '../artifacts/plots/class_accuracy.png'
                    )
                    
                    st.success("✅ Gráficos salvos em `artifacts/plots/`")
                except Exception as e:
                    st.error(f"❌ Erro ao salvar: {e}")
                    import traceback
                    with st.expander("Detalhes do erro"):
                        st.code(traceback.format_exc())

# ============================================
# Seção 4: Inferência
# ============================================
st.header("🔮 4. Inferência")

if st.session_state.model is None or st.session_state.vectorizer is None:
    st.warning("⚠️ Treine o modelo primeiro!")
else:
    inference_mode = st.radio(
        "Modo de inferência:",
        ["Texto único", "Múltiplos textos"],
        horizontal=True
    )
    
    if inference_mode == "Texto único":
        text_input = st.text_area(
            "Digite o texto para classificar:",
            height=150,
            placeholder="Digite aqui o texto que deseja classificar..."
        )
        
        if st.button("🔍 Classificar Texto", use_container_width=True):
            if text_input.strip():
                try:
                    from deploy import Predictor
                    
                    model_path = f'../models/{st.session_state.config.get("representation", "tfidf")}_model.pt'
                    
                    if not os.path.exists(model_path):
                        st.error(f"❌ Modelo não encontrado em {model_path}. Treine o modelo primeiro!")
                    elif not os.path.exists('../artifacts/vectorizer.joblib'):
                        st.error("❌ Vectorizer não encontrado. Treine o modelo primeiro!")
                    else:
                        predictor = Predictor(
                            model_path=model_path,
                            vectorizer_path='../artifacts/vectorizer.joblib',
                            device=st.session_state.device
                        )
                        
                        label, confidence = predict_text(predictor, text_input)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Classe Predita", label)
                        with col2:
                            st.metric("Confiança", f"{confidence:.2%}")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao classificar: {str(e)}")
                    import traceback
                    with st.expander("Detalhes do erro"):
                        st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Digite um texto para classificar!")
    
    else:
        texts_input = st.text_area(
            "Digite os textos (um por linha):",
            height=200,
            placeholder="Texto 1\nTexto 2\nTexto 3\n..."
        )
        
        if st.button("🔍 Classificar Textos", use_container_width=True):
            if texts_input.strip():
                try:
                    from deploy import Predictor
                    
                    texts = [t.strip() for t in texts_input.split('\n') if t.strip()]
                    
                    model_path = f'../models/{st.session_state.config.get("representation", "tfidf")}_model.pt'
                    
                    if not os.path.exists(model_path):
                        st.error(f"❌ Modelo não encontrado em {model_path}. Treine o modelo primeiro!")
                    elif not os.path.exists('../artifacts/vectorizer.joblib'):
                        st.error("❌ Vectorizer não encontrado. Treine o modelo primeiro!")
                    else:
                        predictor = Predictor(
                            model_path=model_path,
                            vectorizer_path='../artifacts/vectorizer.joblib',
                            device=st.session_state.device
                        )
                        
                        results = []
                        progress_bar = st.progress(0)
                        for i, text in enumerate(texts):
                            label, confidence = predict_text(predictor, text)
                            results.append({
                                'Texto': text[:100] + '...' if len(text) > 100 else text,
                                'Classe': label,
                                'Confiança': f"{confidence:.2%}"
                            })
                            progress_bar.progress((i + 1) / len(texts))
                        
                        df_results = pd.DataFrame(results)
                        st.dataframe(df_results, use_container_width=True, hide_index=True)
                        
                        # Botão para salvar
                        if st.button("💾 Salvar Resultados", use_container_width=True, key="save_results"):
                            try:
                                os.makedirs('../reports', exist_ok=True)
                                with open('../reports/example_inference.txt', 'w', encoding='utf-8') as f:
                                    f.write("=== Exemplo de Inferência - TextTorch ===\n\n")
                                    for i, result in enumerate(results, 1):
                                        f.write(f"--- Exemplo {i} ---\n")
                                        f.write(f"Texto: '{result['Texto']}'\n")
                                        f.write(f"  → Predição: {result['Classe']}\n")
                                        f.write(f"  → Confiança: {result['Confiança']}\n\n")
                                st.success("✅ Resultados salvos em `reports/example_inference.txt`")
                            except Exception as e:
                                st.error(f"❌ Erro ao salvar: {e}")
                
                except Exception as e:
                    st.error(f"❌ Erro ao classificar: {str(e)}")
                    import traceback
                    with st.expander("Detalhes do erro"):
                        st.code(traceback.format_exc())
            else:
                st.warning("⚠️ Digite pelo menos um texto para classificar!")

# Footer
st.markdown("---")
st.markdown("**TextTorch Explorer** - Interface interativa para classificação de texto")
