"""
Deployment module for TextTorch.
Provides a simple function to predict the class of a given text.
"""

import os
import logging
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import joblib

# Importar módulos do projeto
from preprocessing import clean_text
from representation import TFIDFRepresentation, sparse_to_dense_tensors
from model import load_model

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Predictor:
    """
    Classe para carregar artefatos e realizar predições.
    """
    
    def __init__(self, model_path: str = 'models/tfidf_model.pt', 
                 vectorizer_path: str = 'artifacts/vectorizer.joblib',
                 device: str = 'cpu'):
        """
        Inicializa o preditor.
        
        Args:
            model_path: Caminho para o modelo salvo
            vectorizer_path: Caminho para o vectorizer salvo
            device: Dispositivo ('cpu' ou 'cuda')
        """
        logger.info("Inicializando preditor...")
        
        # Carrega modelo
        self.model = load_model(model_path, device=device)
        self.device = device
        
        # Carrega vectorizer
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"ERRO: Vectorizer não encontrado em '{vectorizer_path}'")
        # Carrega o vectorizer (TfidfVectorizer do scikit-learn)
        self.vectorizer = joblib.load(vectorizer_path)
        
        # Carrega metadados do checkpoint do modelo
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Carrega mapeamento de labels (se disponível)
        self.n_classes = checkpoint.get('n_classes', 20) # Default para 20 Newsgroups
        
        # Tenta carregar idx_to_label do checkpoint
        if 'idx_to_label' in checkpoint:
            self.idx_to_label = checkpoint['idx_to_label']
        else:
            # Se não estiver no checkpoint, cria um genérico
            self.idx_to_label = {i: f"Classe {i}" for i in range(self.n_classes)}
        
        logger.info("Preditor inicializado com sucesso.")

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Prediz a classe de um texto.
        
        Args:
            text: Texto de entrada
            
        Returns:
            Tupla (label_predita, probabilidade)
        """
        if not text or not isinstance(text, str):
            raise ValueError("ERRO: Texto de entrada deve ser uma string não vazia.")
        
        self.model.eval()
        
        # 1. Preprocessamento
        cleaned_text = clean_text(text)
        
        # 2. Representação
        # O vectorizer espera uma lista de textos
        vectorized_text = self.vectorizer.transform([cleaned_text])
        
        # Converte para tensor denso
        tensor = sparse_to_dense_tensors(vectorized_text)
        tensor = tensor.to(self.device)
        
        # 3. Predição
        with torch.no_grad():
            outputs = self.model(tensor)
            probas = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probas, 1)
        
        # 4. Mapeia para label
        predicted_label = self.idx_to_label.get(predicted_idx.item(), "Classe Desconhecida")
        
        return predicted_label, confidence.item()

    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """
        Prediz a classe para um batch de textos.
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de tuplas (label_predita, probabilidade)
        """
        return [self.predict(text) for text in texts]


def save_inference_example(predictor: Predictor, example_texts: List[str], 
                           save_path: str = 'reports/example_inference.txt'):
    """
    Salva um exemplo de inferência em arquivo.
    
    Args:
        predictor: Instância do preditor
        example_texts: Lista de textos de exemplo
        save_path: Caminho para salvar o arquivo
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=== Exemplo de Inferência - TextTorch ===\n\n")
        
        for i, text in enumerate(example_texts):
            label, prob = predictor.predict(text)
            
            f.write(f"--- Exemplo {i+1} ---\n")
            f.write(f"Texto: '{text[:100]}...'\n")
            f.write(f"  → Predição: {label}\n")
            f.write(f"  → Confiança: {prob:.2%}\n\n")
    
    logger.info(f"Exemplo de inferência salvo em: {save_path}")


if __name__ == "__main__":
    # Teste rápido (requer modelo e vectorizer treinados)
    print("Testando módulo de deploy...")
    
    MODEL_PATH = 'models/tfidf_model.pt'
    VECTORIZER_PATH = 'artifacts/vectorizer.joblib'
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("\nAVISO: Modelo ou vectorizer não encontrados.")
        print("Execute os notebooks de treinamento primeiro para testar o deploy.")
    else:
        try:
            # Inicializa preditor
            predictor = Predictor(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH)
            
            # Textos de exemplo
            texts_to_predict = [
                "This is a test about computer graphics and GPU performance.",
                "The government announced new policies for space exploration.",
                "O futebol brasileiro precisa de mais investimentos na base."
            ]
            
            # Prediz
            predictions = predictor.predict_batch(texts_to_predict)
            
            for text, (label, prob) in zip(texts_to_predict, predictions):
                print(f"Texto: '{text[:50]}...' -> Predição: {label} ({prob:.2%})")
            
            # Salva exemplo
            save_inference_example(predictor, texts_to_predict)
            
        except Exception as e:
            print(f"\nOcorreu um erro durante o teste: {e}")
            print("Certifique-se de que os artefatos salvos são compatíveis com o código atual.")
