#!/usr/bin/env python3
"""
Download the trained model for evaluation
"""

import os
import logging
from pathlib import Path
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_model():
    """Setup the model for evaluation."""
    logger.info("Setting up model for evaluation...")
    
    try:
        # Your Hugging Face repo name
        model_repo = "Shadow56/custom-pdf-heading-extractor"
        
        logger.info(f"Downloading trained model from Hugging Face: {model_repo}")
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_repo, trust_remote_code=True)
        model = DistilBertForSequenceClassification.from_pretrained(model_repo, trust_remote_code=True)
        
        # Save locally
        model_dir = Path("student_final")
        model_dir.mkdir(exist_ok=True)
        
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        
        logger.info("✅ Model setup complete!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model setup failed: {e}")
        # Fallback to base model
        logger.info("Using base DistilBERT as fallback...")
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=4,
            id2label={0: 'O', 1: 'H1', 2: 'H2', 3: 'H3'},
            label2id={'O': 0, 'H1': 1, 'H2': 2, 'H3': 3}
        )
        
        model_dir = Path("student_final")
        model_dir.mkdir(exist_ok=True)
        tokenizer.save_pretrained(model_dir)
        model.save_pretrained(model_dir)
        return True

if __name__ == "__main__":
    setup_model()
