"""Utility functions for PDF span extraction and JSON helpers"""
import json
import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TextSpan:
    """Represents a text span from PDF with metadata."""
    text: str
    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_name: str
    is_bold: bool
    is_centered: bool
    span_id: str  # Unique identifier

def extract_layout_blocks(pdf_path: str) -> List[TextSpan]:
    """
    Extracts all individual text spans from a PDF, keeping them fragmented.
    This provides the raw data needed for the merging function.
    """
    spans = []
    span_idx = 0
    pdf_stem = Path(pdf_path).stem
    
    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                page_width = page.rect.width
                page_height = page.rect.height
                
                blocks = page.get_text("dict", flags=11)["blocks"]
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    
                    block_y0 = block['bbox'][1]
                    # Header/Footer Filtering
                    if block_y0 < page_height * 0.08 or block_y0 > page_height * 0.92:
                        continue

                    for line in block["lines"]:
                        if not line['spans']: continue
                        
                        for span_data in line['spans']:
                            text = span_data['text'].strip()
                            if len(text) < 1 or re.match(r'^[._\-\s]+$', text):
                                continue

                            font_size = span_data["size"]
                            font_name = span_data["font"]
                            is_bold = bool(span_data["flags"] & 2**4) or 'bold' in font_name.lower()
                            
                            bbox = fitz.Rect(span_data['bbox'])
                            center_x = (bbox.x0 + bbox.x1) / 2
                            is_centered = abs(center_x - page_width/2) < page_width * 0.1
                            
                            text_span = TextSpan(
                                text=text,
                                page_num=page_num,
                                bbox=tuple(bbox),
                                font_size=font_size,
                                font_name=font_name,
                                is_bold=is_bold,
                                is_centered=is_centered,
                                span_id=f"{pdf_stem}_{page_num}_{span_idx}"
                            )
                            spans.append(text_span)
                            span_idx += 1
    except Exception as e:
        logger.error(f"Error extracting spans from {pdf_path}: {e}")
    
    return spans

def merge_adjacent_heading_fragments(spans: List[TextSpan]) -> List[TextSpan]:
    """
    FIXED: Merges adjacent text spans that form a single logical heading.
    """
    if not spans:
        return []

    merged_spans = []
    i = 0
    while i < len(spans):
        current_span = spans[i]

        # Look ahead to see if the next span should be merged with this one
        if (i + 1) < len(spans):
            next_span = spans[i + 1]

            # --- Merge Conditions ---
            # 1. They are on the same page.
            # 2. They are on the same visual line (y-coordinates are very close).
            # 3. They are horizontally next to each other (small gap).
            # 4. They share a similar font style (size and boldness).
            is_on_same_line = (current_span.page_num == next_span.page_num and 
                               abs(current_span.bbox[1] - next_span.bbox[1]) < 5)
            
            is_horizontally_adjacent = (next_span.bbox[0] - current_span.bbox[2]) < 15 # Allow up to 15px gap
            
            has_similar_style = (abs(current_span.font_size - next_span.font_size) < 1 and
                                 current_span.is_bold == next_span.is_bold)

            if is_on_same_line and is_horizontally_adjacent and has_similar_style:
                # Merge them into a new span
                combined_text = f"{current_span.text} {next_span.text}"
                combined_bbox = tuple(fitz.Rect(current_span.bbox) | fitz.Rect(next_span.bbox))

                merged_span = TextSpan(
                    text=combined_text,
                    page_num=current_span.page_num,
                    bbox=combined_bbox,
                    font_size=current_span.font_size,
                    font_name=current_span.font_name,
                    is_bold=current_span.is_bold,
                    is_centered=False, # Merged spans are unlikely to be perfectly centered
                    span_id=current_span.span_id + "_merged"
                )
                
                # Replace the current span with the new merged one and remove the next one
                # We do this by modifying the list in place and continuing the loop
                spans[i] = merged_span
                del spans[i + 1]
                continue # Re-evaluate the new merged span with the next one in the list

        # If no merge happened, add the current span to our results and move on
        merged_spans.append(current_span)
        i += 1
    
    return merged_spans

# --- Other utility functions remain the same ---

def normalize_text(text: str) -> str:
    return ' '.join(text.split()).strip().rstrip('.,;:')

def text_similarity(text1: str, text2: str) -> float:
    text1_norm = normalize_text(text1.lower())
    text2_norm = normalize_text(text2.lower())
    if not text1_norm or not text2_norm: return 0.0
    if text1_norm == text2_norm: return 1.0
    if text1_norm in text2_norm or text2_norm in text1_norm: return 0.9
    words1 = set(text1_norm.split())
    words2 = set(text2_norm.split())
    if not words1 or not words2: return 0.0
    return len(words1 & words2) / len(words1 | words2)

def load_annotations(json_path: str) -> Dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_jsonl(data: List[Dict], output_path: str):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_jsonl(input_path: str) -> List[Dict]:
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data
