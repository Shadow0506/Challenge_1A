#!/usr/bin/env python3
"""
PDF Heading Extraction - Challenge 1A
Process PDFs and extract headings using machine learning model
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Import local modules
from pdf_parser import PDFParser
from heading_extractor import HeadingExtractor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def process_single_pdf(pdf_path: str, heading_extractor: HeadingExtractor, pdf_parser: PDFParser) -> Dict[str, Any]:
    """Process a single PDF file and return results."""
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    try:
        # Extract text spans with layout information
        spans = pdf_parser.extract_spans(pdf_path)
        
        # Extract headings using ML model
        headings = heading_extractor.extract_headings(spans)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Get total pages (approximate from spans)
        total_pages = max([span.get('page_num', 0) for span in spans], default=0) + 1
        
        # Format output according to schema
        result = {
            "filename": Path(pdf_path).name,
            "headings": [
                {
                    "text": h["text"],
                    "level": h["label"],
                    "page": h.get("page_num", 0),
                    "confidence": h.get("confidence", 1.0),
                    "bbox": h.get("bbox", [0, 0, 0, 0])
                }
                for h in headings
            ],
            "processing_time": round(processing_time, 3),
            "total_pages": total_pages
        }
        
        logger.info(f"Processed {Path(pdf_path).name}: {len(headings)} headings, {processing_time:.2f}s")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Failed to process {Path(pdf_path).name}: {e}")
        
        return {
            "filename": Path(pdf_path).name,
            "headings": [],
            "processing_time": round(processing_time, 3),
            "total_pages": 0,
            "error": str(e)
        }


def main():
    """Main processing function."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Docker container paths
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # For local testing, allow command line arguments
    if len(sys.argv) == 3:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find model
    model_path = "student_final"
    if not os.path.exists(model_path):
        logger.error("Model not found. Please ensure student_final model is available.")
        sys.exit(1)
    
    logger.info(f"Loading model from: {model_path}")
    
    # Initialize components
    pdf_parser = PDFParser()
    heading_extractor = HeadingExtractor(model_path)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    total_start_time = time.time()
    for pdf_file in pdf_files:
        try:
            # Process the PDF
            result = process_single_pdf(str(pdf_file), heading_extractor, pdf_parser)
            
            # Save output
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Critical error processing {pdf_file.name}: {e}")
    
    total_time = time.time() - total_start_time
    logger.info(f"Completed processing {len(pdf_files)} files in {total_time:.2f}s")


if __name__ == "__main__":
    main()
