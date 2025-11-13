"""
Verification script for the TextTorch project.
Checks if all required files and directories exist and meet the project specifications.
"""

import os
from pathlib import Path
import sys

# --- Configuration ---
BASE_DIR = Path(__file__).parent.resolve()
REQUIRED_DIRS = [
    "raw",
    "notebooks",
    "src",
    "models",
    "artifacts",
    "artifacts/plots",
    "reports",
    "reports/figures",
    "slides",
]
REQUIRED_FILES = [
    "requirements.txt",
    "README.md",
    "models/config.yaml",
    "slides/PytorchExplorer_Seminar.pptx",
]
SRC_MODULES = [
    "preprocessing.py",
    "representation.py",
    "model.py",
    "train.py",
    "evaluate.py",
    "deploy.py",
]
NOTEBOOKS = [
    "01_preprocessing.ipynb",
    "02_representation.ipynb",
    "03_model_definition.ipynb",
    "04_training.ipynb",
    "05_evaluation.ipynb",
    "06_deploy.ipynb",
]
# --- End Configuration ---

class Verifier:
    """Class to run verification checks."""

    def __init__(self):
        self.errors = 0
        self.warnings = 0

    def check(self, condition, message, is_warning=False):
        """Runs a single check."""
        if not condition:
            if is_warning:
                print(f"  [!] AVISO: {message}")
                self.warnings += 1
            else:
                print(f"  [X] ERRO: {message}")
                self.errors += 1
            return False
        return True

    def run(self):
        """Runs all verification checks."""
        print("--- Verificando a integridade do projeto TextTorch ---\n")

        # 1. Verificar Estrutura de Diretórios
        print("1. Verificando estrutura de diretórios...")
        for d in REQUIRED_DIRS:
            dir_path = BASE_DIR / d
            self.check(dir_path.is_dir(), f"Diretório '{d}' não encontrado.")
        print("   ... Estrutura de diretórios OK.\n")

        # 2. Verificar Arquivos Essenciais
        print("2. Verificando arquivos essenciais...")
        for f in REQUIRED_FILES:
            file_path = BASE_DIR / f
            self.check(file_path.is_file(), f"Arquivo '{f}' não encontrado.")
        print("   ... Arquivos essenciais OK.\n")

        # 3. Verificar Módulos `src`
        print("3. Verificando módulos em 'src/'...")
        for m in SRC_MODULES:
            module_path = BASE_DIR / "src" / m
            self.check(module_path.is_file(), f"Módulo '{m}' não encontrado em 'src/'.")
        print("   ... Módulos OK.\n")

        # 4. Verificar Notebooks
        print("4. Verificando notebooks em 'notebooks/'...")
        for n in NOTEBOOKS:
            notebook_path = BASE_DIR / "notebooks" / n
            self.check(notebook_path.is_file(), f"Notebook '{n}' não encontrado em 'notebooks/'.")
        print("   ... Notebooks OK.\n")
        
        # 5. Verificar Dataset Customizado (Aviso)
        print("5. Verificando dataset customizado (opcional)...")
        csv_path = BASE_DIR / "raw" / "Base_dados_textos_6_classes.csv"
        self.check(csv_path.is_file(), 
                   "Arquivo 'Base_dados_textos_6_classes.csv' não encontrado em 'raw/'. O pipeline funcionará apenas com o 20 Newsgroups.",
                   is_warning=True)
        print("   ... Verificação de dataset concluída.\n")

        # --- Relatório Final ---
        print("--- Verificação Concluída ---")
        if self.errors == 0:
            print(f"✅ Sucesso! Todos os {len(REQUIRED_DIRS) + len(REQUIRED_FILES) + len(SRC_MODULES) + len(NOTEBOOKS)} itens obrigatórios foram encontrados.")
            if self.warnings > 0:
                print(f"⚠️  Atenção: {self.warnings} aviso(s) foram emitidos (verifique acima).")
            return True
        else:
            print(f"❌ Falha! {self.errors} erro(s) e {self.warnings} aviso(s) encontrados.")
            print("   Por favor, corrija os erros listados acima para garantir a execução correta do projeto.")
            return False

if __name__ == "__main__":
    verifier = Verifier()
    success = verifier.run()
    
    # Retorna código de saída para uso em scripts
    if not success:
        sys.exit(1)
