"""
Training module for TextTorch.
Handles model training, validation, and checkpointing.
"""

import os
import time
import logging
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Define seeds para reprodutibilidade."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeds definidas: {seed}")


def get_device() -> str:
    """Detecta e retorna o dispositivo disponível (GPU ou CPU)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Dispositivo detectado: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_dataloader(X: torch.Tensor, y: torch.Tensor, batch_size: int = 32, 
                      shuffle: bool = True) -> DataLoader:
    """
    Cria DataLoader a partir de tensores.
    
    Args:
        X: Features (tensores)
        y: Labels (tensores)
        batch_size: Tamanho do batch
        shuffle: Se True, embaralha os dados
        
    Returns:
        DataLoader
    """
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class Trainer:
    """Classe para gerenciar treinamento do modelo."""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', learning_rate: float = 1e-3,
                 weight_decay: float = 0.0001):
        """
        Inicializa o trainer.
        
        Args:
            model: Modelo PyTorch
            device: Dispositivo ('cpu' ou 'cuda')
            learning_rate: Taxa de aprendizado
            weight_decay: Weight decay para regularização L2
        """
        self.model = model.to(device)
        self.device = device
        
        # Loss e otimizador
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Histórico de treinamento
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        logger.info(f"Trainer inicializado: lr={learning_rate}, weight_decay={weight_decay}")
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Treina uma época.
        
        Args:
            dataloader: DataLoader de treinamento
            
        Returns:
            Tupla (loss média, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in tqdm(dataloader, desc="Treinamento"):
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Métricas
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Valida o modelo.
        
        Args:
            dataloader: DataLoader de validação
            
        Returns:
            Tupla (loss média, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Métricas
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: int = 10, early_stopping: bool = True, patience: int = 3,
              save_path: str = 'models/tfidf_model.pt', idx_to_label: Optional[Dict] = None) -> Dict:
        """
        Loop completo de treinamento.
        
        Args:
            train_loader: DataLoader de treinamento
            val_loader: DataLoader de validação (opcional)
            epochs: Número de épocas
            early_stopping: Se True, usa early stopping
            patience: Paciência para early stopping
            save_path: Caminho para salvar o melhor modelo
            
        Returns:
            Dicionário com histórico de treinamento
        """
        logger.info(f"Iniciando treinamento: {epochs} épocas")
        start_time = time.time()
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(1, epochs + 1):
            # Treina uma época
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validação
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                logger.info(
                    f"Época {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
                
                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        # Salva melhor modelo
                        self.save_model(save_path, idx_to_label=idx_to_label)
                        logger.info(f"  → Melhor modelo salvo (val_loss: {val_loss:.4f})")
                    else:
                        epochs_without_improvement += 1
                        
                    if epochs_without_improvement >= patience:
                        logger.info(f"Early stopping acionado após {epoch} épocas")
                        break
            else:
                logger.info(
                    f"Época {epoch}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
                )
                
                # Salva modelo sem validação
                if epoch == epochs:
                    self.save_model(save_path, idx_to_label=idx_to_label)
        
        # Tempo total
        elapsed_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {elapsed_time/60:.2f} minutos")
        
        # Retorna histórico
        history = {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'epochs_trained': len(self.train_losses),
            'training_time': elapsed_time
        }
        
        return history
    
    def save_model(self, path: str, idx_to_label: Optional[Dict] = None):
        """Salva checkpoint do modelo."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        # Salva metadados do modelo
        if hasattr(self.model, 'input_dim'):
            checkpoint['input_dim'] = self.model.input_dim
        if hasattr(self.model, 'n_classes'):
            checkpoint['n_classes'] = self.model.n_classes
        if hasattr(self.model, 'hidden_dim'):
            checkpoint['hidden_dim'] = self.model.hidden_dim
        
        checkpoint['model_class'] = self.model.__class__.__name__
        
        # Salva mapeamento de labels se fornecido
        if idx_to_label is not None:
            checkpoint['idx_to_label'] = idx_to_label
        
        torch.save(checkpoint, path)


def smoke_test(model: nn.Module, X_sample: torch.Tensor, y_sample: torch.Tensor,
               device: str = 'cpu', n_epochs: int = 1) -> Dict:
    """
    Smoke test: treina em pequena amostra para verificar funcionamento.
    
    Args:
        model: Modelo PyTorch
        X_sample: Amostra de features
        y_sample: Amostra de labels
        device: Dispositivo
        n_epochs: Número de épocas
        
    Returns:
        Dicionário com resultados
    """
    logger.info("=== SMOKE TEST ===")
    logger.info(f"Amostra: {X_sample.shape[0]} documentos, {n_epochs} época(s)")
    
    # Cria dataloader
    dataloader = create_dataloader(X_sample, y_sample, batch_size=16, shuffle=True)
    
    # Treina
    trainer = Trainer(model, device=device, learning_rate=1e-3)
    history = trainer.train(dataloader, val_loader=None, epochs=n_epochs, early_stopping=False)
    
    # Testa predição
    model.eval()
    with torch.no_grad():
        outputs = model(X_sample[:5].to(device))
        _, predictions = torch.max(outputs, 1)
        
    logger.info(f"Predições de exemplo: {predictions.cpu().numpy()}")
    logger.info(f"Labels reais: {y_sample[:5].numpy()}")
    logger.info(f"Loss final: {history['train_losses'][-1]:.4f}")
    logger.info("=== SMOKE TEST CONCLUÍDO ===\n")
    
    return history


if __name__ == "__main__":
    # Teste rápido
    print("Testando módulo de treinamento...")
    
    from src.model import TFIDFClassifier
    
    # Dados dummy
    X_train = torch.randn(100, 500)
    y_train = torch.randint(0, 5, (100,))
    X_val = torch.randn(20, 500)
    y_val = torch.randint(0, 5, (20,))
    
    # Modelo
    model = TFIDFClassifier(input_dim=500, n_classes=5)
    
    # DataLoaders
    train_loader = create_dataloader(X_train, y_train, batch_size=16)
    val_loader = create_dataloader(X_val, y_val, batch_size=16, shuffle=False)
    
    # Treina
    device = get_device()
    trainer = Trainer(model, device=device)
    history = trainer.train(train_loader, val_loader, epochs=3, early_stopping=False)
    
    print(f"Treinamento concluído: {history['epochs_trained']} épocas")
