"""
Preprocessing module for TextTorch.
Handles data loading, cleaning, and preparation for text classification.
"""

import os
import re
import random
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Define seed para reprodutibilidade.
    
    Args:
        seed: Valor da seed
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Seeds definidas: {seed}")


def load_20newsgroups(subset: str = 'all', remove_headers: bool = True) -> Tuple[List[str], List[int], List[str]]:
    """
    Carrega o dataset 20 Newsgroups.
    
    Args:
        subset: 'train', 'test' ou 'all'
        remove_headers: Se True, remove headers/footers/quotes
        
    Returns:
        texts: Lista de textos
        labels: Lista de labels (índices)
        label_names: Lista de nomes das classes
    """
    logger.info(f"Carregando 20 Newsgroups (subset={subset})...")
    
    # Remove cabeçalhos, rodapés e citações para evitar overfitting
    remove = ('headers', 'footers', 'quotes') if remove_headers else ()
    
    if subset == 'all':
        # Carrega train e test e combina
        train_data = fetch_20newsgroups(subset='train', remove=remove, shuffle=True, random_state=42)
        test_data = fetch_20newsgroups(subset='test', remove=remove, shuffle=True, random_state=42)
        
        texts = list(train_data.data) + list(test_data.data)
        labels = list(train_data.target) + list(test_data.target)
        label_names = list(train_data.target_names)
    else:
        data = fetch_20newsgroups(subset=subset, remove=remove, shuffle=True, random_state=42)
        texts = list(data.data)
        labels = list(data.target)
        label_names = list(data.target_names)
    
    logger.info(f"20 Newsgroups carregado: {len(texts)} documentos, {len(label_names)} classes")
    return texts, labels, label_names


def load_custom_csv(csv_path: str, text_col: str = 'text', label_col: str = 'label') -> Tuple[List[str], List[str]]:
    """
    Carrega dataset customizado de um arquivo CSV.
    
    Args:
        csv_path: Caminho para o arquivo CSV
        text_col: Nome da coluna contendo os textos
        label_col: Nome da coluna contendo os labels
        
    Returns:
        texts: Lista de textos
        labels: Lista de labels (strings)
        
    Raises:
        FileNotFoundError: Se o arquivo não existir
        ValueError: Se as colunas necessárias não existirem
    """
    logger.info(f"Carregando CSV customizado: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ERRO: Arquivo CSV não encontrado em '{csv_path}'.\n"
            f"Por favor, coloque o arquivo 'Base_dados_textos_6_classes.csv' na pasta 'raw/'."
        )
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        # Tenta outros encodings comuns
        df = pd.read_csv(csv_path, encoding='latin-1')
    
    # Valida colunas
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"ERRO: CSV deve conter as colunas '{text_col}' e '{label_col}'.\n"
            f"Colunas encontradas: {list(df.columns)}"
        )
    
    # Remove linhas com valores nulos
    df = df.dropna(subset=[text_col, label_col])
    
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    
    logger.info(f"CSV carregado: {len(texts)} documentos, {len(set(labels))} classes únicas")
    return texts, labels


def clean_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    """
    Limpa e normaliza texto.
    
    Args:
        text: Texto a ser limpo
        lowercase: Se True, converte para minúsculas
        remove_punct: Se True, remove pontuação
        
    Returns:
        Texto limpo
    """
    # Remove caracteres especiais e múltiplos espaços
    text = re.sub(r'\s+', ' ', text)
    
    if remove_punct:
        # Remove pontuação, mantém apenas letras e números
        text = re.sub(r'[^\w\s]', ' ', text)
    
    if lowercase:
        text = text.lower()
    
    # Remove espaços extras
    text = text.strip()
    
    return text


def preprocess_texts(texts: List[str], max_length: Optional[int] = None) -> List[str]:
    """
    Preprocessa lista de textos.
    
    Args:
        texts: Lista de textos crus
        max_length: Comprimento máximo de caracteres (opcional)
        
    Returns:
        Lista de textos processados
    """
    logger.info(f"Preprocessando {len(texts)} textos...")
    
    processed = []
    for text in texts:
        # Limpa o texto
        clean = clean_text(text, lowercase=True, remove_punct=True)
        
        # Trunca se necessário
        if max_length and len(clean) > max_length:
            clean = clean[:max_length]
        
        processed.append(clean)
    
    logger.info("Preprocessamento concluído")
    return processed


def prepare_dataset(
    dataset_choice: str = 'custom_csv',
    csv_path: str = 'raw/Base_dados_textos_6_classes.csv',
    csv_text_col: str = 'text',
    csv_label_col: str = 'label',
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Prepara dataset completo para treinamento.
    
    Args:
        dataset_choice: '20newsgroups' ou 'custom_csv' (use apenas um por vez)
        csv_path: Caminho para CSV customizado
        csv_text_col: Nome da coluna de texto no CSV
        csv_label_col: Nome da coluna de label no CSV
        test_size: Proporção do conjunto de teste
        val_size: Proporção do conjunto de validação
        seed: Seed para reprodutibilidade
        
    Returns:
        Dicionário com splits de dados e metadados
    """
    set_seed(seed)
    
    # Valida escolha do dataset
    if dataset_choice not in ['20newsgroups', 'custom_csv']:
        raise ValueError(
            f"dataset_choice deve ser '20newsgroups' ou 'custom_csv', não '{dataset_choice}'. "
            "Use apenas um dataset por vez."
        )
    
    all_texts = []
    all_labels = []
    
    # Carrega 20 Newsgroups
    if dataset_choice == '20newsgroups':
        texts_20n, labels_20n, label_names_20n = load_20newsgroups(subset='all')
        all_texts.extend(texts_20n)
        # Usa os nomes das classes diretamente, sem prefixo
        all_labels.extend([label_names_20n[l] for l in labels_20n])
        logger.info(f"20 Newsgroups carregado: {len(texts_20n)} documentos")
    
    # Carrega CSV customizado
    elif dataset_choice == 'custom_csv':
        texts_csv, labels_csv = load_custom_csv(
            csv_path, text_col=csv_text_col, label_col=csv_label_col
        )
        all_texts.extend(texts_csv)
        # Usa os labels diretamente, sem prefixo
        all_labels.extend([str(l) for l in labels_csv])
        logger.info(f"CSV customizado carregado: {len(texts_csv)} documentos")
    
    # Preprocessa textos
    processed_texts = preprocess_texts(all_texts)
    
    # Cria mapeamento de labels para índices
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    # Converte labels para índices
    label_indices = [label_to_idx[l] for l in all_labels]
    
    # Split: train/temp -> train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        processed_texts, label_indices, test_size=(test_size + val_size), random_state=seed, stratify=label_indices
    )
    
    # Split temp em val e test
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), random_state=seed, stratify=y_temp
    )
    
    dataset = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'n_classes': len(unique_labels),
        'class_names': unique_labels
    }
    
    logger.info(f"Dataset preparado:")
    logger.info(f"  Train: {len(X_train)} documentos")
    logger.info(f"  Val: {len(X_val)} documentos")
    logger.info(f"  Test: {len(X_test)} documentos")
    logger.info(f"  Classes: {len(unique_labels)}")
    
    return dataset


if __name__ == "__main__":
    # Teste rápido
    print("Testando módulo de preprocessing...")
    dataset = prepare_dataset(dataset_choice='20newsgroups', test_size=0.2, val_size=0.1)
    print(f"Classes: {dataset['n_classes']}")
    print(f"Exemplo de texto: {dataset['X_train'][0][:100]}...")
