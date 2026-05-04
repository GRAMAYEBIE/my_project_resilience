import logging
import sys
from pathlib import Path
from src.config import paths

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure un système de logging double : Console + Fichier.
    Niveau de production : Gère la rotation et les formats pro.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Format détaillé pour le debugging
    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Handler pour la Console (Terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)

    # 2. Handler pour le Fichier (Log persistant)
    log_file = paths.logs_dir / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)

    # Configuration du logger racine
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # On nettoie les handlers existants pour éviter les doublons
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.info(f"🚀 Logging initialisé [Niveau: {level}]")
    logging.info(f"📁 Fichier de log : {log_file}")
    
    return logger

