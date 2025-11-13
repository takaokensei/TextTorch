"""
Representation module for TextTorch.
Handles text representation using TF-IDF or embeddings.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TFIDFRepresentation:
    """
    Classe para representação TF-IDF.
    Converte textos em vetores TF-IDF e salva/carrega o vectorizer.
    """
    
    def __init__(self, min_df: int = 5, max_df: float = 0.8, 
                 ngram_range: Tuple[int, int] = (1, 2), max_features: int = 10000):
        """
        Inicializa o vectorizer TF-IDF.
        
        Args:
            min_df: Frequência mínima de documento (ignora termos muito raros)
            max_df: Frequência máxima de documento (ignora termos muito comuns)
            ngram_range: Range de n-gramas (1,2) = unigramas e bigramas
            max_features: Número máximo de features
        """
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            max_features=max_features,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        self.is_fitted = False
        
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Treina o vectorizer e transforma os textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Matriz TF-IDF (sparse)
        """
        logger.info(f"Treinando TF-IDF em {len(texts)} documentos...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        
        logger.info(f"TF-IDF treinado: {tfidf_matrix.shape[1]} features")
        logger.info(f"Esparsidade: {(1.0 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2%}")
        
        return tfidf_matrix
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforma textos usando vectorizer já treinado.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Matriz TF-IDF (sparse)
        """
        if not self.is_fitted:
            raise ValueError("ERRO: Vectorizer não foi treinado. Use fit_transform() primeiro.")
        
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self) -> List[str]:
        """Retorna lista de features (palavras/n-gramas)."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        """
        Retorna as top N features por importância média.
        
        Args:
            n: Número de features a retornar
            
        Returns:
            Lista de tuplas (feature, score)
        """
        if not self.is_fitted:
            return []
        
        feature_names = self.get_feature_names()
        # Calcula importância média de cada feature
        idf_scores = self.vectorizer.idf_
        top_indices = np.argsort(idf_scores)[-n:][::-1]
        
        return [(feature_names[i], idf_scores[i]) for i in top_indices]
    
    def save(self, path: str):
        """Salva o vectorizer treinado."""
        if not self.is_fitted:
            logger.warning("Vectorizer não foi treinado, salvando mesmo assim...")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info(f"Vectorizer salvo em: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TFIDFRepresentation':
        """
        Carrega vectorizer de arquivo.
        
        Args:
            path: Caminho do arquivo
            
        Returns:
            Instância de TFIDFRepresentation
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"ERRO: Vectorizer não encontrado em '{path}'")
        
        representation = cls()
        representation.vectorizer = joblib.load(path)
        representation.is_fitted = True
        logger.info(f"Vectorizer carregado de: {path}")
        
        return representation
    
    @property
    def input_dim(self) -> int:
        """Retorna dimensão do vetor de entrada."""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.get_feature_names_out())


def sparse_to_dense_tensors(sparse_matrix: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    """
    Converte matriz esparsa (scipy) para tensor denso do PyTorch.
    
    Args:
        sparse_matrix: Matriz esparsa do scipy
        dtype: Tipo de dado do tensor
        
    Returns:
        Tensor denso do PyTorch
    """
    # Converte para array denso
    dense_array = sparse_matrix.toarray()
    # Converte para tensor PyTorch
    tensor = torch.tensor(dense_array, dtype=dtype)
    
    return tensor


# ============================================
# OPCIONAL: REPRESENTAÇÃO COM EMBEDDINGS
# ============================================
# Esta seção está comentada e pode ser ativada quando necessário.
# Para usar embeddings:
# 1. Altere representation: embedding no config.yaml
# 2. Descomente o código abaixo
# 3. Reexecute o notebook 02_representation.ipynb

"""
class EmbeddingRepresentation:
    '''
    Classe para representação com embeddings treináveis ou pré-treinados.
    '''
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 300):
        '''
        Inicializa representação com embeddings.
        
        Args:
            vocab_size: Tamanho do vocabulário
            embedding_dim: Dimensão dos embeddings
        '''
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.is_fitted = False
        
    def build_vocabulary(self, texts: List[str], min_freq: int = 5):
        '''
        Constrói vocabulário a partir dos textos.
        
        Args:
            texts: Lista de textos
            min_freq: Frequência mínima para incluir palavra no vocabulário
        '''
        logger.info(f"Construindo vocabulário de {len(texts)} documentos...")
        
        from collections import Counter
        
        # Conta palavras
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        # Filtra por frequência e ordena
        vocab = [word for word, count in word_counts.items() if count >= min_freq]
        vocab = sorted(vocab)[:self.vocab_size - 2]  # -2 para PAD e UNK
        
        # Cria mapeamentos (PAD=0, UNK=1)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.word_to_idx.update({word: idx + 2 for idx, word in enumerate(vocab)})
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        self.vocab_size = len(self.word_to_idx)
        self.is_fitted = True
        
        logger.info(f"Vocabulário construído: {self.vocab_size} palavras")
    
    def text_to_sequence(self, text: str, max_length: int = 200) -> List[int]:
        '''
        Converte texto para sequência de índices.
        
        Args:
            text: Texto de entrada
            max_length: Comprimento máximo da sequência
            
        Returns:
            Lista de índices
        '''
        words = text.split()[:max_length]
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 = UNK
        
        # Padding
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))  # 0 = PAD
            
        return indices
    
    def texts_to_sequences(self, texts: List[str], max_length: int = 200) -> torch.Tensor:
        '''
        Converte lista de textos para tensor de sequências.
        
        Args:
            texts: Lista de textos
            max_length: Comprimento máximo da sequência
            
        Returns:
            Tensor de forma (batch_size, max_length)
        '''
        if not self.is_fitted:
            raise ValueError("ERRO: Vocabulário não foi construído. Use build_vocabulary() primeiro.")
        
        sequences = [self.text_to_sequence(text, max_length) for text in texts]
        return torch.tensor(sequences, dtype=torch.long)
    
    def save(self, path: str):
        '''Salva vocabulário e configurações.'''
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(data, path)
        logger.info(f"Vocabulário salvo em: {path}")
    
    @classmethod
    def load(cls, path: str) -> 'EmbeddingRepresentation':
        '''
        Carrega vocabulário de arquivo.
        
        Args:
            path: Caminho do arquivo
            
        Returns:
            Instância de EmbeddingRepresentation
        '''
        if not os.path.exists(path):
            raise FileNotFoundError(f"ERRO: Vocabulário não encontrado em '{path}'")
        
        data = joblib.load(path)
        
        representation = cls(vocab_size=data['vocab_size'], embedding_dim=data['embedding_dim'])
        representation.word_to_idx = data['word_to_idx']
        representation.idx_to_word = data['idx_to_word']
        representation.is_fitted = data['is_fitted']
        
        logger.info(f"Vocabulário carregado de: {path}")
        return representation
"""


if __name__ == "__main__":
    # Teste rápido
    print("\n=== Testando TF-IDF ===")
    texts = [
        "this is a test document",
        "another test document here",
        "one more document for testing"
    ]
    
    tfidf = TFIDFRepresentation(min_df=1, max_df=1.0)
    matrix = tfidf.fit_transform(texts)
    print(f"Shape: {matrix.shape}")
    print(f"Features: {tfidf.get_feature_names()[:10]}")
    
    # Converte para tensor
    tensor = sparse_to_dense_tensors(matrix)
    print(f"Tensor shape: {tensor.shape}")
