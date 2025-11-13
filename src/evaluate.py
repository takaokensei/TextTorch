"""
Evaluation module for TextTorch.
Handles model evaluation, metrics calculation, and visualization.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurar estilo de plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = 'cpu',
                   class_names: Optional[List[str]] = None) -> Dict:
    """
    Avalia modelo no conjunto de dados.
    
    Args:
        model: Modelo PyTorch
        dataloader: DataLoader
        device: Dispositivo
        class_names: Nomes das classes (opcional)
        
    Returns:
        Dicionário com métricas
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probas = []
    
    logger.info("Avaliando modelo...")
    
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            probas = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probas = np.array(all_probas)
    
    # Calcula métricas
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    
    precision_micro = precision_score(all_labels, all_predictions, average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_predictions, average='micro', zero_division=0)
    f1_micro = f1_score(all_labels, all_predictions, average='micro', zero_division=0)
    
    # Métricas por classe
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    # Matriz de confusão
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Relatório de classificação
    if class_names:
        report = classification_report(all_labels, all_predictions, 
                                      target_names=class_names, zero_division=0)
    else:
        report = classification_report(all_labels, all_predictions, zero_division=0)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_micro': float(precision_micro),
        'recall_micro': float(recall_micro),
        'f1_micro': float(f1_micro),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_predictions.tolist(),
        'labels': all_labels.tolist(),
        'probabilities': all_probas.tolist()
    }
    
    logger.info(f"Avaliação concluída:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  F1-Score (macro): {f1_macro:.4f}")
    logger.info(f"  F1-Score (micro): {f1_micro:.4f}")
    
    return metrics


def find_misclassified_examples(metrics: Dict, texts: List[str], 
                                 class_names: List[str], top_n: int = 10) -> List[Dict]:
    """
    Encontra exemplos classificados incorretamente.
    
    Args:
        metrics: Dicionário de métricas
        texts: Lista de textos
        class_names: Nomes das classes
        top_n: Número de exemplos a retornar
        
    Returns:
        Lista de exemplos com maior confiança incorreta
    """
    predictions = np.array(metrics['predictions'])
    labels = np.array(metrics['labels'])
    probas = np.array(metrics['probabilities'])
    
    # Encontra índices incorretos
    incorrect_idx = np.where(predictions != labels)[0]
    
    if len(incorrect_idx) == 0:
        logger.warning("Nenhum exemplo classificado incorretamente!")
        return []
    
    # Pega confiança da predição incorreta
    incorrect_confidences = probas[incorrect_idx, predictions[incorrect_idx]]
    
    # Ordena por confiança (maior confiança = erro mais "confiante")
    sorted_indices = np.argsort(incorrect_confidences)[::-1][:top_n]
    
    misclassified = []
    for idx in sorted_indices:
        original_idx = incorrect_idx[idx]
        misclassified.append({
            'text': texts[original_idx][:200] + '...' if len(texts[original_idx]) > 200 else texts[original_idx],
            'true_label': class_names[labels[original_idx]],
            'predicted_label': class_names[predictions[original_idx]],
            'confidence': float(probas[original_idx, predictions[original_idx]])
        })
    
    return misclassified


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: str = 'artifacts/plots/confusion_matrix.png'):
    """
    Plota matriz de confusão.
    
    Args:
        cm: Matriz de confusão
        class_names: Nomes das classes
        save_path: Caminho para salvar figura
    """
    plt.figure(figsize=(12, 10))
    
    # Normaliza para porcentagens
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plot com seaborn
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Porcentagem (%)'})
    
    plt.title('Matriz de Confusão', fontsize=16, fontweight='bold')
    plt.ylabel('Classe Real', fontsize=12)
    plt.xlabel('Classe Predita', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Matriz de confusão salva em: {save_path}")


def plot_training_curves(history: Dict, save_path: str = 'artifacts/plots/training_curves.png'):
    """
    Plota curvas de treinamento (loss e accuracy).
    
    Args:
        history: Histórico de treinamento
        save_path: Caminho para salvar figura
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    # Plot Loss
    ax1.plot(epochs, history['train_losses'], 'b-', label='Treino', linewidth=2)
    if history['val_losses']:
        ax1.plot(epochs, history['val_losses'], 'r-', label='Validação', linewidth=2)
    ax1.set_title('Loss ao Longo das Épocas', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, history['train_accs'], 'b-', label='Treino', linewidth=2)
    if history['val_accs']:
        ax2.plot(epochs, history['val_accs'], 'r-', label='Validação', linewidth=2)
    ax2.set_title('Acurácia ao Longo das Épocas', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=12)
    ax2.set_ylabel('Acurácia', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Curvas de treinamento salvas em: {save_path}")


def plot_class_accuracy(metrics: Dict, class_names: List[str],
                       save_path: str = 'artifacts/plots/class_accuracy.png'):
    """
    Plota acurácia por classe.
    
    Args:
        metrics: Dicionário de métricas
        class_names: Nomes das classes
        save_path: Caminho para salvar figura
    """
    f1_scores = metrics['f1_per_class']
    
    # Ordena por F1-score
    sorted_indices = np.argsort(f1_scores)
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_classes)))
    bars = plt.barh(range(len(sorted_classes)), sorted_f1, color=colors)
    
    plt.yticks(range(len(sorted_classes)), sorted_classes)
    plt.xlabel('F1-Score', fontsize=12)
    plt.title('F1-Score por Classe', fontsize=14, fontweight='bold')
    plt.xlim(0, 1.0)
    
    # Adiciona valores nas barras
    for i, (bar, score) in enumerate(zip(bars, sorted_f1)):
        plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico de acurácia por classe salvo em: {save_path}")


def save_metrics(metrics: Dict, save_path: str = 'artifacts/metrics.json'):
    """
    Salva métricas em arquivo JSON.
    
    Args:
        metrics: Dicionário de métricas
        save_path: Caminho para salvar
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Remove dados grandes para o JSON (mantém apenas métricas agregadas)
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
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Métricas salvas em: {save_path}")


if __name__ == "__main__":
    # Teste rápido
    print("Testando módulo de avaliação...")
    
    # Dados dummy
    cm = np.array([[80, 10, 5], [15, 75, 10], [5, 20, 75]])
    class_names = ['Classe A', 'Classe B', 'Classe C']
    
    # Plot
    plot_confusion_matrix(cm, class_names, save_path='test_cm.png')
    print("Plot de teste criado")
