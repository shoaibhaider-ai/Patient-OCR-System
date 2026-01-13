
# Patient Data Extraction OCR System

CV Lab Computer Vision Project - 2024

## Description
Web-based OCR system that automatically extracts patient information from medical documents with 85-95% accuracy.

### Extracted Fields (5)
1. Patient Name
2. Address
3. Contact Information
4. Insurance ID
5. Diagnosis

## Technology Stack
- **Backend:** Python, FastAPI, Tesseract OCR, OpenCV
- **Frontend:** React
- **Features:** Ensemble pattern matching, Fuzzy matching, Field validation

## Prerequisites
- Python 3.8+
- Node.js 14+
- Tesseract OCR ([Download](https://github.com/UB-Mannheim/tesseract/wiki))

## Installation

### 1. Install Tesseract OCR
Download and install to: `C:\Program Files\Tesseract-OCR`

### 2. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

## Running the System

### Option 1: Using Scripts (Recommended)
Double-click `run-system.bat`

### Option 2: Manual Start
**Terminal 1 - Backend:**
```bash
cd backend
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

## Access URLs
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## Testing
1. Open `test-patient-simple.html` in browser
2. Take screenshot (Win+Shift+S)
3. Upload to http://localhost:3000
4. Click "Extract Data"
5. View extracted fields

## System Features
✓ Ensemble pattern matching with confidence scoring  
✓ Fuzzy string matching for OCR error correction  
✓ Field validation and filtering  
✓ Medical term recognition  
✓ Position-aware extraction  
✓ Multi-OCR support (Tesseract + optional EasyOCR)

## Expected Accuracy
- Clean documents: 90-100%
- Moderate quality: 80-95%
- Poor quality: 70-85%

## Project Structure
```
patient-ocr-system/
├── backend/
│   ├── app.py              # Main FastAPI application
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.js         # React main component
│   │   └── App.css        # Styling
│   └── package.json       # Node.js dependencies
├── sample-patient-document.html
├── test-patient-simple.html
├── run-system.bat
└── README.md
```

## CV Lab Project - 2024
