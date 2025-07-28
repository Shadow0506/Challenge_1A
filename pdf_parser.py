"""
PDF Parser - Extracts paragraphs with formatting information
Implements steps 1-2 of the algorithm
"""

import fitz
import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Paragraph:
    """Represents a paragraph with formatting info."""
    text: str
    page_num: int
    position: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_name: str
    is_bold: bool
    is_centered: bool
    starts_with_number: bool
    paragraph_index: int


class PDFParser:
    """Extracts formatted paragraphs from PDF."""
    
    def __init__(self, config):
        self.config = config
        
    def extract_paragraphs(self, pdf_path: str) -> List[Paragraph]:
        """Extract paragraphs with formatting from PDF."""
        paragraphs = []
        
        try:
            with fitz.open(pdf_path) as doc:
                paragraph_index = 0
                
                for page_num, page in enumerate(doc, 1):
                    # Get page dimensions
                    page_width = page.rect.width
                    
                    # Extract blocks
                    blocks = page.get_text("dict", flags=11)
                    
                    # Process each block as potential paragraph
                    for block in blocks["blocks"]:
                        if "lines" not in block:
                            continue
                        
                        # Combine lines into paragraph
                        para_text = []
                        font_sizes = []
                        font_names = []
                        font_flags = []
                        bbox = None
                        
                        for line in block["lines"]:
                            line_text = []
                            
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text.append(text)
                                    font_sizes.append(span["size"])
                                    font_names.append(span["font"])
                                    font_flags.append(span["flags"])
                                    
                                    # Update bounding box
                                    span_bbox = span["bbox"]
                                    if bbox is None:
                                        bbox = list(span_bbox)
                                    else:
                                        bbox[0] = min(bbox[0], span_bbox[0])
                                        bbox[1] = min(bbox[1], span_bbox[1])
                                        bbox[2] = max(bbox[2], span_bbox[2])
                                        bbox[3] = max(bbox[3], span_bbox[3])
                            
                            if line_text:
                                para_text.append(" ".join(line_text))
                        
                        if para_text and bbox:
                            # Create paragraph
                            full_text = " ".join(para_text)
                            
                            # Skip very short text
                            if len(full_text.strip()) < 2:
                                continue
                            
                            # Determine properties
                            avg_font_size = sum(font_sizes) / len(font_sizes)
                            main_font = max(set(font_names), key=font_names.count)
                            is_bold = any(f & 2**4 for f in font_flags)
                            
                            # Check if centered
                            center_x = (bbox[0] + bbox[2]) / 2
                            is_centered = abs(center_x - page_width/2) < page_width * 0.1
                            
                            # Check if starts with number
                            starts_with_number = bool(re.match(
                                r'^[\s]*(\d+\.?|[IVXLCDM]+\.?|[a-zA-Z]\.)\s+', 
                                full_text
                            ))
                            
                            para = Paragraph(
                                text=full_text,
                                page_num=page_num,
                                position=tuple(bbox),
                                font_size=avg_font_size,
                                font_name=main_font,
                                is_bold=is_bold,
                                is_centered=is_centered,
                                starts_with_number=starts_with_number,
                                paragraph_index=paragraph_index
                            )
                            
                            paragraphs.append(para)
                            paragraph_index += 1
                
        except Exception as e:
            logger.error(f"Error parsing PDF: {str(e)}")
        
        return paragraphs