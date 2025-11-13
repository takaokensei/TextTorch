"""
Utility functions for Streamlit app.
Wrappers for src/ modules to integrate with Streamlit interface.
"""

import os
import sys
import yaml
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Adiciona src ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import prepare_dataset, set_seed
from representation import TFIDFRepresentation, sparse_to_dense_tensors
from model import load_config as _load_config, create_model
from train import Trainer, get_device, create_dataloader
from evaluate import evaluate_model, find_misclassified_examples
from deploy import Predictor


def load_and_prepare_dataset(
    dataset_choice: str,
    csv_path: str = '../raw/Base_dados_textos_6_classes.csv',
    csv_text_col: str = 'text',
    csv_label_col: str = 'label',
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Wrapper para prepare_dataset do módulo preprocessing.
    
    Args:
        dataset_choice: '20newsgroups' ou 'custom_csv'
        csv_path: Caminho para CSV customizado
        csv_text_col: Nome da coluna de texto no CSV
        csv_label_col: Nome da coluna de label no CSV
        test_size: Proporção do conjunto de teste
        val_size: Proporção do conjunto de validação
        seed: Seed para reprodutibilidade
        
    Returns:
        Dicionário com splits de dados e metadados
    """
    return prepare_dataset(
        dataset_choice=dataset_choice,
        csv_path=csv_path,
        csv_text_col=csv_text_col,
        csv_label_col=csv_label_col,
        test_size=test_size,
        val_size=val_size,
        seed=seed
    )


def create_vectorizer(config: Dict) -> TFIDFRepresentation:
    """
    Cria vectorizer TF-IDF baseado na configuração.
    
    Args:
        config: Dicionário de configuração
        
    Returns:
        Instância de TFIDFRepresentation
    """
    return TFIDFRepresentation(
        min_df=config.get('min_df', 5),
        max_df=config.get('max_df', 0.8),
        ngram_range=tuple(config.get('ngram_range', [1, 2])),
        max_features=config.get('max_features', 10000)
    )


def vectorize_texts(
    vectorizer: TFIDFRepresentation,
    texts: List[str],
    fit: bool = False
) -> torch.Tensor:
    """
    Transforma textos em tensores usando vectorizer.
    
    Args:
        vectorizer: Instância de TFIDFRepresentation
        texts: Lista de textos
        fit: Se True, treina o vectorizer primeiro
        
    Returns:
        Tensor PyTorch com representações
    """
    if fit:
        sparse_matrix = vectorizer.fit_transform(texts)
    else:
        sparse_matrix = vectorizer.transform(texts)
    
    return sparse_to_dense_tensors(sparse_matrix)


def create_model_from_config(
    config: Dict,
    input_dim: int,
    n_classes: int
) -> nn.Module:
    """
    Cria modelo baseado na configuração.
    
    Args:
        config: Dicionário de configuração
        input_dim: Dimensão de entrada
        n_classes: Número de classes
        
    Returns:
        Modelo PyTorch
    """
    return create_model(input_dim=input_dim, n_classes=n_classes, config=config)


def train_model_wrapper(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: Dict,
    idx_to_label: Optional[Dict] = None
) -> Dict:
    """
    Wrapper para treinamento do modelo.
    
    Args:
        model: Modelo PyTorch
        train_loader: DataLoader de treinamento
        val_loader: DataLoader de validação (opcional)
        config: Dicionário de configuração
        idx_to_label: Mapeamento de índices para labels (opcional)
        
    Returns:
        Dicionário com histórico de treinamento
    """
    device = get_device()
    
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=config.get('learning_rate', 0.001),
        weight_decay=config.get('weight_decay', 0.0001)
    )
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.get('epochs', 10),
        early_stopping=config.get('early_stopping', True),
        patience=config.get('patience', 3),
        save_path=config.get('model_save_path', '../models/tfidf_model.pt'),
        idx_to_label=idx_to_label
    )
    
    return history


def evaluate_model_wrapper(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    class_names: List[str]
) -> Dict:
    """
    Wrapper para avaliação do modelo.
    
    Args:
        model: Modelo PyTorch
        dataloader: DataLoader
        device: Dispositivo ('cpu' ou 'cuda')
        class_names: Nomes das classes
        
    Returns:
        Dicionário com métricas
    """
    return evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        class_names=class_names
    )


def predict_text(
    predictor: Predictor,
    text: str
) -> Tuple[str, float]:
    """
    Wrapper para predição de texto.
    
    Args:
        predictor: Instância de Predictor
        text: Texto para classificar
        
    Returns:
        Tupla (label_predita, confiança)
    """
    return predictor.predict(text)


def load_config(config_path: str = '../models/config.yaml') -> Dict:
    """
    Wrapper para carregar configuração.
    
    Args:
        config_path: Caminho do arquivo de configuração
        
    Returns:
        Dicionário com configurações
    """
    return _load_config(config_path)


def save_config(config: Dict, config_path: str = '../models/config.yaml'):
    """
    Salva configuração em arquivo YAML.
    
    Args:
        config: Dicionário de configuração
        config_path: Caminho para salvar
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_device_wrapper() -> str:
    """
    Wrapper para detectar dispositivo.
    
    Returns:
        'cpu' ou 'cuda'
    """
    return get_device()


def find_misclassified(
    metrics: Dict,
    texts: List[str],
    class_names: List[str],
    top_n: int = 10
) -> List[Dict]:
    """
    Wrapper para encontrar exemplos mal classificados.
    
    Args:
        metrics: Dicionário de métricas
        texts: Lista de textos
        class_names: Nomes das classes
        top_n: Número de exemplos a retornar
        
    Returns:
        Lista de exemplos mal classificados
    """
    return find_misclassified_examples(metrics, texts, class_names, top_n)

