"""
Production heading extractor using distilled student model
"""

import logging
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from utils import extract_layout_blocks, TextSpan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL2ID = {'O': 0, 'H1': 1, 'H2': 2, 'H3': 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class HeadingExtractor:
    """Production heading extractor using distilled model."""
    
    def __init__(self, model_path: str = "student_final"):
        """Initialize with student model."""
        logger.info(f"Loading model from {model_path}")
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        # Try to load quantized model if available
        quantized_path = Path(model_path) / "quantized_model.pt"
        if quantized_path.exists():
            logger.info("Loading quantized model")
            try:
                import torch.quantization as quantization
                self.model = quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                self.model.load_state_dict(torch.load(quantized_path))
            except:
                logger.warning("Failed to load quantized model, using standard model")
        
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.batch_size = 32
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """Extract outline from PDF using model."""
        # Extract spans
        spans = extract_layout_blocks(pdf_path)
        
        if not spans:
            return {"title": "", "outline": []}
        
        # Predict labels for all spans
        predictions = self._batch_predict(spans)
        
        # Build outline
        title = ""
        outline = []
        
        for span, pred_label in zip(spans, predictions):
            if pred_label == 'O':
                continue
            
            # First H1 on page 1 might be title
            if not title and pred_label == 'H1' and span.page_num == 0:
                title = span.text.strip()
            else:
                outline.append({
                    "level": pred_label,
                    "text": span.text.strip(),
                    "page": span.page_num + 1  # Convert to 1-based
                })
        
        # Post-process outline
        outline = self._post_process_outline(outline)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _batch_predict(self, spans: List[TextSpan]) -> List[str]:
        """Batch prediction for efficiency."""
        texts = [span.text for span in spans]
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt"
                ).to(self.device)
                
                # Predict
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Get predictions
                batch_preds = torch.argmax(logits, dim=-1)
                
                # Convert to labels
                for pred_id in batch_preds:
                    predictions.append(ID2LABEL[pred_id.item()])
        
        return predictions
    
    def _post_process_outline(self, outline: List[Dict]) -> List[Dict]:
        """Clean up and ensure logical hierarchy."""
        if not outline:
            return outline
        
        # Remove duplicates
        seen = set()
        cleaned = []
        
        for item in outline:
            key = (item['text'], item['page'])
            if key not in seen:
                seen.add(key)
                cleaned.append(item)
        
        # Ensure logical hierarchy
        processed = []
        has_h1 = False
        has_h2 = False
        
        for item in cleaned:
            level = item['level']
            
            # Adjust levels if needed
            if level == 'H3' and not has_h2:
                item['level'] = 'H2'
            elif level == 'H2' and not has_h1:
                item['level'] = 'H1'
            
            # Update flags
            if item['level'] == 'H1':
                has_h1 = True
                has_h2 = False
            elif item['level'] == 'H2':
                has_h2 = True
            
            processed.append(item)
        
        return processed


# For backward compatibility with existing pipeline
def extract_headings_from_pdf(pdf_path: str, model_path: str = "student_final") -> Dict:
    """Convenience function for extraction."""
    extractor = HeadingExtractor(model_path)
    return extractor.extract_outline(pdf_path)