"""
Model module for TextTorch.
Defines PyTorch neural network architectures for text classification.
"""

import os
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import yaml

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TFIDFClassifier(nn.Module):
    """
    Classificador feedforward simples para vetores TF-IDF.
    Arquitetura: Input → Linear(512) → ReLU → Dropout → Linear(n_classes)
    """
    
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 512, dropout: float = 0.5):
        """
        Inicializa o modelo.
        
        Args:
            input_dim: Dimensão do vetor de entrada (TF-IDF)
            n_classes: Número de classes
            hidden_dim: Dimensão da camada oculta
            dropout: Taxa de dropout
        """
        super(TFIDFClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        
        # Camadas
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)
        
        logger.info(f"TFIDFClassifier criado: input={input_dim}, hidden={hidden_dim}, output={n_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Tensor de entrada (batch_size, input_dim)
            
        Returns:
            Logits (batch_size, n_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def count_parameters(self) -> int:
        """Conta o número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================
# OPCIONAL: MODELO COM EMBEDDINGS
# ============================================
# Esta seção está comentada e pode ser ativada quando necessário.
# Para usar embeddings:
# 1. Altere representation: embedding no config.yaml
# 2. Descomente o código abaixo
# 3. Reexecute os notebooks de modelo e treinamento

"""
class EmbeddingClassifier(nn.Module):
    '''
    Classificador com embeddings treináveis + LSTM.
    Arquitetura: Embedding → LSTM → Linear(n_classes)
    '''
    
    def __init__(self, vocab_size: int, embedding_dim: int, n_classes: int, 
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.5,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        '''
        Inicializa o modelo com embeddings.
        
        Args:
            vocab_size: Tamanho do vocabulário
            embedding_dim: Dimensão dos embeddings
            n_classes: Número de classes
            hidden_dim: Dimensão oculta do LSTM
            num_layers: Número de camadas LSTM
            dropout: Taxa de dropout
            pretrained_embeddings: Embeddings pré-treinados (opcional)
        '''
        super(EmbeddingClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        
        # Camada de embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Carrega embeddings pré-treinados se fornecidos
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            logger.info("Embeddings pré-treinados carregados")
        
        # LSTM
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Camada de saída (2 * hidden_dim porque LSTM é bidirecional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        
        logger.info(f"EmbeddingClassifier criado: vocab={vocab_size}, embed={embedding_dim}, "
                   f"hidden={hidden_dim}, output={n_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.
        
        Args:
            x: Tensor de índices de palavras (batch_size, seq_length)
            
        Returns:
            Logits (batch_size, n_classes)
        '''
        # Embedding: (batch_size, seq_length) → (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)
        
        # LSTM: (batch_size, seq_length, embedding_dim) → (batch_size, seq_length, hidden_dim*2)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Usa o último estado oculto (concatena forward e backward)
        # hidden: (num_layers*2, batch_size, hidden_dim) → (batch_size, hidden_dim*2)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        # Dropout + Linear
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        
        return logits
    
    def count_parameters(self) -> int:
        '''Conta o número total de parâmetros treináveis.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
"""


def load_config(config_path: str = 'models/config.yaml') -> Dict:
    """
    Carrega configurações do arquivo YAML.
    
    Args:
        config_path: Caminho do arquivo de configuração
        
    Returns:
        Dicionário com configurações
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"ERRO: Arquivo de configuração não encontrado em '{config_path}'")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuração carregada de: {config_path}")
    return config


def create_model(input_dim: int, n_classes: int, config: Optional[Dict] = None) -> nn.Module:
    """
    Factory function para criar modelo baseado na configuração.
    
    Args:
        input_dim: Dimensão de entrada
        n_classes: Número de classes
        config: Dicionário de configuração (opcional)
        
    Returns:
        Modelo PyTorch
    """
    if config is None:
        config = load_config()
    
    representation = config.get('representation', 'tfidf')
    hidden_dim = config.get('hidden_dim', 512)
    dropout = config.get('dropout', 0.5)
    
    if representation == 'tfidf':
        model = TFIDFClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    else:
        raise NotImplementedError(
            f"Representação '{representation}' não implementada.\n"
            f"Para usar embeddings, descomente a classe EmbeddingClassifier em src/model.py"
        )
    
    return model


def save_model(model: nn.Module, path: str, config: Optional[Dict] = None):
    """
    Salva modelo treinado.
    
    Args:
        model: Modelo PyTorch
        path: Caminho para salvar
        config: Configuração do modelo (opcional)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Salva estado do modelo e metadados
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
    }
    
    if config:
        checkpoint['config'] = config
    
    if hasattr(model, 'input_dim'):
        checkpoint['input_dim'] = model.input_dim
    if hasattr(model, 'n_classes'):
        checkpoint['n_classes'] = model.n_classes
    
    torch.save(checkpoint, path)
    logger.info(f"Modelo salvo em: {path}")


def load_model(path: str, device: str = 'cpu') -> nn.Module:
    """
    Carrega modelo salvo.
    
    Args:
        path: Caminho do modelo
        device: Dispositivo ('cpu' ou 'cuda')
        
    Returns:
        Modelo carregado
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"ERRO: Modelo não encontrado em '{path}'")
    
    checkpoint = torch.load(path, map_location=device)
    
    model_class = checkpoint.get('model_class', 'TFIDFClassifier')
    input_dim = checkpoint.get('input_dim')
    n_classes = checkpoint.get('n_classes')
    config = checkpoint.get('config', {})
    
    # Recria o modelo
    if model_class == 'TFIDFClassifier':
        model = TFIDFClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=config.get('hidden_dim', 512),
            dropout=config.get('dropout', 0.5)
        )
    else:
        raise ValueError(f"Classe de modelo '{model_class}' não reconhecida")
    
    # Carrega pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Modelo carregado de: {path}")
    return model


if __name__ == "__main__":
    # Teste rápido
    print("\n=== Testando TFIDFClassifier ===")
    model = TFIDFClassifier(input_dim=5000, n_classes=20, hidden_dim=512, dropout=0.5)
    print(f"Parâmetros: {model.count_parameters():,}")
    
    # Teste forward pass
    dummy_input = torch.randn(8, 5000)  # batch_size=8, input_dim=5000
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # esperado: (8, 20)
