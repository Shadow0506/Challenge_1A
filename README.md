# Challenge 1A - PDF Heading Extraction

## Overview
This solution uses a distilled BERT model to extract and classify headings (H1, H2, H3) from PDF documents using layout and text features.

## Solution Architecture
- **Model**: DistilBERT-base-uncased (distilled from BERT teacher model)
- **Features**: Text content + layout tokens (bold, font size, position, centering)
- **Classes**: H1, H2, H3 (Other text is filtered out)
- **Performance**: Optimized for 10-second processing of 50-page PDFs

## Directory Structure
```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # JSON files provided as outputs
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Docker container configuration
├── process_pdfs.py      # Main processing script
├── pdf_parser.py        # PDF text extraction
├── heading_extractor.py # ML-based heading classification
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── student_final/       # Trained DistilBERT model
└── README.md           # This file
```

## Usage

### Build the Docker image
```bash
docker build --platform linux/amd64 -t pdf-processor .
```

### Test with sample data
```bash
docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/input:ro -v $(pwd)/sample_dataset/outputs:/app/output --network none pdf-processor
```

### Local development
```bash
python process_pdfs.py input_directory output_directory
```

## Output Format
Each PDF generates a JSON file with the following structure:
```json
{
  "filename": "example.pdf",
  "headings": [
    {
      "text": "Introduction",
      "level": "H1",
      "page": 0,
      "confidence": 0.95,
      "bbox": [100, 200, 300, 220]
    }
  ],
  "processing_time": 2.5,
  "total_pages": 10
}
```

## Performance Specifications
- **Processing Speed**: < 10 seconds for 50-page PDFs
- **Memory Usage**: < 16GB RAM limit
- **Architecture**: AMD64 compatible
- **Network**: Works completely offline
- **Model Size**: Optimized DistilBERT (~250MB)

## Technical Features
- **Layout-Aware**: Uses font size, boldness, positioning features
- **Hierarchical Training**: Trained with penalties for H1↔H2 confusion
- **Span Merging**: Intelligently combines split headings
- **No Hardcoding**: Generic solution for any PDF structure
- **Error Handling**: Graceful failure with empty output for problematic PDFs

## Validation Checklist
- ✅ All PDFs in input directory are processed
- ✅ JSON output files are generated for each PDF
- ✅ Output format matches required structure
- ✅ Output conforms to schema in sample_dataset/schema/output_schema.json
- ✅ Processing completes within 10 seconds for 50-page PDFs
- ✅ Solution works without internet access
- ✅ Memory usage stays within 16GB limit
- ✅ Compatible with AMD64 architecture

## Model Training Details
- **Teacher Model**: BERT-base-uncased with hierarchical loss (Macro F1: 0.758)
- **Student Model**: Knowledge distillation to DistilBERT for production deployment
- **Training Data**: Gold-labeled dataset with layout feature engineering
- **Feature Engineering**: Bold, font size, position, and centering tokens
