from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import re
import pytesseract
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
from fuzzywuzzy import fuzz
from collections import Counter

# Configure Tesseract path
possible_tesseract_paths = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    r'C:\Tesseract-OCR\tesseract.exe',
]

for path in possible_tesseract_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

# Try to import EasyOCR (optional)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Patient Data Extraction OCR System - Ultimate Edition")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader if available
reader = None
if EASYOCR_AVAILABLE:
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        logger.info("EasyOCR reader initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize EasyOCR: {e}")
        EASYOCR_AVAILABLE = False

class UltimateOCRProcessor:
    def __init__(self):
        self.reader = reader
        # Medical terms database for diagnosis
        self.medical_terms = [
            'diabetes', 'mellitus', 'hypertension', 'asthma', 'cancer', 
            'flu', 'influenza', 'covid', 'pneumonia', 'bronchitis',
            'arthritis', 'allergy', 'depression', 'anxiety', 'migraine'
        ]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Advanced image preprocessing"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return morph
    
    def extract_text_tesseract(self, image: np.ndarray) -> str:
        """Extract text using Tesseract OCR"""
        try:
            text1 = pytesseract.image_to_string(image, config='--psm 6')
            preprocessed = self.preprocess_image(image)
            text2 = pytesseract.image_to_string(preprocessed, config='--psm 6')
            return text1 if len(text1) > len(text2) else text2
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return ""
    
    def extract_text_easyocr(self, image: np.ndarray) -> str:
        """Extract text using EasyOCR"""
        if not EASYOCR_AVAILABLE or reader is None:
            return ""
        try:
            results1 = self.reader.readtext(image, detail=0, paragraph=True)
            text1 = '\n'.join(results1)
            preprocessed = self.preprocess_image(image)
            results2 = self.reader.readtext(preprocessed, detail=0, paragraph=True)
            text2 = '\n'.join(results2)
            return text1 if len(text1) > len(text2) else text2
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return ""
    
    def fuzzy_find_keyword(self, text: str, keywords: List[str], threshold: int = 80) -> Optional[int]:
        """Find keyword position with fuzzy matching to handle OCR errors"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            for word in words:
                if fuzz.ratio(word, keyword_lower) >= threshold:
                    pos = text_lower.find(word)
                    if pos != -1:
                        logger.debug(f"Fuzzy match: '{word}' ~ '{keyword}' at position {pos}")
                        return pos
        return None
    
    def find_field_with_confidence(self, patterns: List[str], text: str, field_name: str) -> Tuple[str, float]:
        """Find field with confidence score - ENSEMBLE APPROACH"""
        candidates = []
        
        for i, pattern in enumerate(patterns):
            try:
                matches = list(re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE))
                for match in matches:
                    extracted = match.group(1).strip()
                    # Calculate confidence
                    confidence = 1.0 - (i * 0.08)  # Pattern priority
                    
                    # Boost for keyword proximity
                    if field_name.lower() in text[max(0, match.start()-30):match.start()].lower():
                        confidence += 0.15
                    
                    # Boost for position in document
                    relative_pos = match.start() / max(len(text), 1)
                    if field_name == "Name" and relative_pos < 0.3:
                        confidence += 0.1
                    elif field_name == "Diagnosis" and relative_pos > 0.5:
                        confidence += 0.1
                    
                    candidates.append((extracted, confidence, i))
                    logger.debug(f"{field_name} candidate: '{extracted[:50]}' (conf: {confidence:.2f})")
            except Exception as e:
                logger.debug(f"Pattern error: {e}")
                continue
        
        if candidates:
            # Sort by confidence and return best
            best = max(candidates, key=lambda x: x[1])
            return best[0], best[1]
        
        return "Not found", 0.0
    
    def validate_name(self, name: str) -> bool:
        """Validate extracted name"""
        if name == "Not found":
            return False
        # Must have 2-4 words
        words = name.split()
        if not (2 <= len(words) <= 4):
            return False
        # Each word should start with capital
        if not all(w[0].isupper() for w in words if w):
            return False
        # No numbers
        if any(char.isdigit() for char in name):
            return False
        # Exclude common false positives
        exclude = ['record', 'medical', 'document', 'hospital', 'patient', 'center', 'health']
        if any(word.lower() in exclude for word in words):
            return False
        return True
    
    def extract_name(self, text: str) -> str:
        """Extract patient name - ENHANCED WITH ENSEMBLE"""
        patterns = [
            # High priority - specific labels
            r'(?:Patient\s+Name|PATIENT\s+NAME)\s*[:\-_=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            r'(?:^|\n)Name\s*[:\-_=]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            
            # Multi-line
            r'(?:Patient\s+Name|PATIENT\s+NAME|Name|NAME)[\s:]*\n\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})',
            
            # With titles
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|DR\.|MR\.|MRS\.|Miss)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',
            
            # Context-aware
            r'(?:Patient|Name).*?\n.*?([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)?)',
            
            # Capitalized names (lowpriority)
            r'\b([A-Z][a-z]{3,}\s+[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{2,})?)\b',
        ]
        
        result, confidence = self.find_field_with_confidence(patterns, text, "Name")
        
        if result != "Not found" and self.validate_name(result):
            result = re.sub(r'\s+', ' ', result)
            logger.info(f"✅ Name extracted: '{result}' (confidence: {confidence:.2f})")
            return result
        
        # Fallback: Look for name-like patterns near "patient" or "name" keywords
        keywords = ['patient', 'name']
        for keyword in keywords:
            pos = self.fuzzy_find_keyword(text, [keyword], threshold=75)
            if pos is not None:
                # Look in next 100 characters
                search_area = text[pos:pos+100]
                match = re.search(r'([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})', search_area)
                if match:
                    candidate = match.group(1)
                    if self.validate_name(candidate):
                        logger.info(f"✅ Name extracted (fuzzy fallback): '{candidate}'")
                        return candidate
        
        logger.warning("❌ Name not found")
        return "Not found"
    
    def extract_address(self, text: str) -> str:
        """Extract address - ENHANCED"""
        patterns = [
            # With street type
            r'(?:Address|Addr|ADDRESS)\s*[:\-_=]\s*([0-9]+[A-Za-z0-9\s,.\-#]+?(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Way|Boulevard|Blvd)[A-Za-z0-9\s,.\-]{0,80})',
            
            # Multi-line
            r'(?:Address|Addr|ADDRESS)[\s:]*\n\s*([0-9][A-Za-z0-9\s,.\-#]{15,120})',
            
            # Number + street
            r'([0-9]{1,6}\s+[A-Z][A-Za-z\s]+?(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln)[A-Za-z0-9\s,.\-]{0,80})',
            
            # With city state ZIP
            r'(?:Address|Addr|ADDRESS)\s*[:\-_=]?\s*(.{15,120}?(?:[A-Z]{2}\s+\d{5}|\b[A-Z][a-z]+,?\s+[A-Z]{2}\b))',
            
            # Just street address
            r'([0-9]{1,6}\s+[A-Za-z][A-Za-z\s,.\-#]{10,80})',
        ]
        
        result, confidence = self.find_field_with_confidence(patterns, text, "Address")
        
        if result != "Not found":
            result = re.sub(r'\s+', ' ', result)
            result = re.sub(r'(?:Contact|Phone|Tel|Insurance|Diagnosis|ID).*$', '', result, flags=re.IGNORECASE).strip()
            
            # Validate: should have numbers and letters
            if re.search(r'\d', result) and len(result) >= 10:
                logger.info(f"✅ Address extracted: '{result}' (confidence: {confidence:.2f})")
                return result
        
        logger.warning("❌ Address not found")
        return "Not found"
    
    def extract_contact(self, text: str) -> str:
        """Extract contact - ENHANCED"""
        phone_patterns = [
            # With keyword
            r'(?:Phone|Tel|Contact|Mobile|Cell|PHONE|CONTACT)\s*[:\-_=]\s*([+]?[0-9\-\(\)\s\.]{10,20})',
            
            # Multi-line
            r'(?:Phone|Contact|PHONE|CONTACT)[\s:]*\n\s*([+]?[0-9\-\(\)\s\.]{10,20})',
            
            # Standard formats
            r'\b(\+?1?[\s\-]?\(?[0-9]{3}\)?[\s\-\.]*[0-9]{3}[\s\-\.]*[0-9]{4})\b',
            r'\b([0-9]{3}[\-\s\.]?[0-9]{3}[\-\s\.]?[0-9]{4})\b',
            
            # International
            r'\b(\+[0-9]{1,3}[\s\-]?[0-9]{3}[\s\-]?[0-9]{3}[\s\-]?[0-9]{4})\b',
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phone = match.group(1).strip()
                phone = re.sub(r'[^\d+\-\(\)\s\.]', '', phone)
                digits = re.sub(r'\D', '', phone)
                if 10 <= len(digits) <= 15:
                    logger.info(f"✅ Contact extracted: '{phone}'")
                    return phone
        
        # Email fallback
        email_pattern = r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
        match = re.search(email_pattern, text)
        if match:
            email = match.group(1).strip()
            logger.info(f"✅ Contact (email) extracted: '{email}'")
            return email
        
        logger.warning("❌ Contact not found")
        return "Not found"
    
    def extract_insurance_id(self, text: str) -> str:
        """Extract insurance ID - ENHANCED WITH VALIDATION"""
        patterns = [
            # With keyword
            r'(?:Insurance\s+ID|Insurance\s+Number|Policy\s+Number|Member\s+ID|INSURANCE\s+ID)\s*[:\-_=]\s*([A-Z0-9\-]{5,20})',
            
            # Multi-line
            r'(?:Insurance\s+ID|INSURANCE\s+ID|Insurance|Policy)[\s:]*\n\s*([A-Z0-9\-]{5,20})',
            
            # Just ID
            r'(?:^|\n)(?:ID|Policy)\s*[:\-_=]\s*([A-Z]{2,4}[0-9]{6,15})',
            
            # Common patterns
            r'\b([A-Z]{2,3}[0-9]{9,12})\b',  # BC123456789
            r'\b([A-Z]{2}[0-9]{6,12}[A-Z]?)\b',
            r'(?:Insurance|Policy|ID)[\s:]+([A-Z]{2,4}[0-9]{6,15})',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                insurance_id = match.group(1).strip().upper()
                # Validate: must have both letters and numbers
                if (re.search(r'[A-Z]', insurance_id) and 
                    re.search(r'[0-9]', insurance_id) and
                    5 <= len(insurance_id) <= 20):
                    logger.info(f"✅ Insurance ID extracted: '{insurance_id}'")
                    return insurance_id
        
        logger.warning("❌ Insurance ID not found")
        return "Not found"
    
    def extract_diagnosis(self, text: str) -> str:
        """Extract diagnosis - ENHANCED WITH MEDICAL TERMS"""
        patterns = [
            # With keyword
            r'(?:Diagnosis|DIAGNOSIS|Condition|Disease)\s*[:\-_=]\s*([A-Za-z][A-Za-z\s,\-]{4,100})',
            
            # Multi-line
            r'(?:Diagnosis|DIAGNOSIS)[\s:]*\n\s*([A-Za-z][A-Za-z\s,\-]{4,100})',
            
            # Diagnosed with
            r'(?:Diagnosed\s+with|Suffers\s+from|Condition:|Disease:)\s+([A-Za-z][A-Za-z\s,\-]{4,100})',
        ]
        
        # Try patterns first
        result, confidence = self.find_field_with_confidence(patterns, text, "Diagnosis")
        
        if result != "Not found":
            result = re.sub(r'\s+', ' ', result)
            result = re.sub(r'[\d:.]+$', '', result).strip()
            result = re.sub(r'\b\d{4}\b.*$', '', result).strip()
            result = re.sub(r'(?:Test|Document|November|December|January|Date|Time|Record).*$', '', result, flags=re.IGNORECASE).strip()
            
            if 3 <= len(result) <= 100:
                logger.info(f"✅ Diagnosis extracted: '{result}' (confidence: {confidence:.2f})")
                return result
        
        # Fallback: Look for medical terms
        text_lower = text.lower()
        found_terms = []
        for term in self.medical_terms:
            if term in text_lower:
                # Extract context around term
                pos = text_lower.find(term)
                context = text[max(0, pos-20):min(len(text), pos+50)]
                found_terms.append(context)
        
        if found_terms:
            # Return most likely diagnosis
            diagnosis = found_terms[0].strip()
            # Clean up
            diagnosis = re.sub(r'^\W+|\W+$', '', diagnosis)
            if len(diagnosis) >= 3:
                logger.info(f"✅ Diagnosis extracted (medical term fallback): '{diagnosis}'")
                return diagnosis
        
        logger.warning("❌ Diagnosis not found")
        return "Not found"
    
    def process_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Main processing pipeline - ULTIMATE VERSION"""
        try:
            # Convert bytes to image
            image = Image.open(io.BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 2:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            elif image_np.shape[2] == 4:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
            elif image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Extract text
            logger.info("🔍 Extracting text with Tesseract...")
            text_tesseract = self.extract_text_tesseract(image_np)
            
            combined_text = text_tesseract
            
            if EASYOCR_AVAILABLE:
                logger.info("🔍 Extracting text with EasyOCR...")
                text_easyocr = self.extract_text_easyocr(image_np)
                combined_text = text_tesseract + "\n" + text_easyocr
            else:
                logger.info("ℹ️  Using Tesseract only (EasyOCR not available)")
            
            logger.info(f"📝 Extracted text length: {len(combined_text)} characters")
            
            # Extract fields with ENHANCED methods
            logger.info("=" * 60)
            logger.info("🎯 FIELD EXTRACTION PHASE")
            logger.info("=" * 60)
            
            result = {
                "name": self.extract_name(combined_text),
                "address": self.extract_address(combined_text),
                "contact": self.extract_contact(combined_text),
                "insurance_id": self.extract_insurance_id(combined_text),
                "diagnosis": self.extract_diagnosis(combined_text),
                "raw_text": combined_text,
                "confidence": self.calculate_confidence({
                    "name": self.extract_name(combined_text),
                    "address": self.extract_address(combined_text),
                    "contact": self.extract_contact(combined_text),
                    "insurance_id": self.extract_insurance_id(combined_text),
                    "diagnosis": self.extract_diagnosis(combined_text),
                })
            }
            
            logger.info("=" * 60)
            logger.info(f"🎯 FINAL CONFIDENCE SCORE: {result['confidence']}%")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def calculate_confidence(self, fields: Dict[str, str]) -> float:
        """Calculate overall confidence score"""
        found_count = sum(1 for v in fields.values() if v != "Not found")
        confidence = (found_count / len(fields)) * 100
        return round(confidence, 2)

# Initialize processor
ocr_processor = UltimateOCRProcessor()

@app.get("/")
async def root():
    return {
        "message": "Patient Data Extraction OCR System - Ultimate Edition",
        "version": "3.0.0 - Ultimate",
        "features": [
            "Ensemble pattern matching",
            "Fuzzy keyword matching",
            "Field validation",
            "Confidence scoring",
            "Medical term recognition",
            "Enhanced logging"
        ],
        "endpoints": ["/extract", "/health"]
    }

@app.post("/extract")
async def extract_patient_data(file: UploadFile = File(...)):
    """Extract patient data - ULTIMATE VERSION"""
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        contents = await file.read()
        logger.info(f"📤 Processing image: {file.filename}")
        result = ocr_processor.process_image(contents)
        
        return JSONResponse(content={
            "success": True,
            "data": {
                "name": result["name"],
                "address": result["address"],
                "contact": result["contact"],
                "insurance_id": result["insurance_id"],
                "diagnosis": result["diagnosis"],
            },
            "confidence": result["confidence"],
            "raw_text": result["raw_text"]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "ultimate",
        "features_enabled": {
            "tesseract": True,
            "easyocr": EASYOCR_AVAILABLE,
            "fuzzy_matching": True,
            "validation": True,
            "ensemble": True
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
