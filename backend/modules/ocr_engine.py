"""
OCR Engine Module for Form Extraction.
Handles text extraction with TrOCR + line segmentation, checkbox detection, and signature detection.
Maintains Tesseract as fallback for compatibility.
"""

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


@dataclass
class TextBox:
    """Represents a detected text region with position and confidence."""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center point of the text box."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class Checkbox:
    """Represents a detected checkbox."""
    x: int
    y: int
    width: int
    height: int
    is_checked: bool
    confidence: float


@dataclass
class OCRResult:
    """Complete OCR result for an image."""
    full_text: str
    text_boxes: List[TextBox]
    checkboxes: List[Checkbox]
    has_signature: bool
    average_confidence: float
    ocr_engine_used: str = "tesseract"  # 'trocr' or 'tesseract'


class OCREngine:
    """
    OCR Engine with TrOCR + Line Segmentation and Tesseract fallback.
    
    Uses SSD-based line segmentation to split documents into lines,
    then processes each line with TrOCR for high-accuracy recognition.
    Falls back to Tesseract if TrOCR is unavailable.
    """
    
    def __init__(
        self,
        lang: str = "eng",
        config: str = "--psm 6",
        use_trocr: bool = None,
        use_tesseract_fallback: bool = True
    ):
        """
        Initialize OCR Engine.
        
        Args:
            lang: Language for OCR (default: English)
            config: Tesseract configuration string
            use_trocr: Whether to use TrOCR (default: from config)
            use_tesseract_fallback: Whether to fallback to Tesseract if TrOCR fails
        """
        self.lang = lang
        self.config = config
        self.use_tesseract_fallback = use_tesseract_fallback
        
        # Determine if TrOCR should be used
        if use_trocr is None:
            trocr_setting = getattr(Config, 'USE_TROCR', 'auto')
            if trocr_setting == 'true':
                self.use_trocr = True
            elif trocr_setting == 'false':
                self.use_trocr = False
            elif trocr_setting == 'auto':
                # Auto mode: only use TrOCR if CUDA is available (too slow on CPU)
                try:
                    import torch
                    self.use_trocr = torch.cuda.is_available()
                    if self.use_trocr:
                        print("TrOCR: CUDA detected, enabling TrOCR")
                    else:
                        print("TrOCR: No CUDA, using Tesseract (TrOCR too slow on CPU)")
                except ImportError:
                    self.use_trocr = False
            else:
                self.use_trocr = False
        else:
            self.use_trocr = use_trocr
        
        # Set Tesseract path if configured
        if Config.TESSERACT_PATH:
            pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
        
        # Lazy-loaded TrOCR components
        self._trocr_engine = None
        self._line_segmentor = None
        self._trocr_initialized = False
        self._trocr_available = None
    
    def _init_trocr(self) -> bool:
        """Initialize TrOCR engine and line segmentor."""
        if self._trocr_initialized:
            return self._trocr_available
        
        self._trocr_initialized = True
        
        if not self.use_trocr:
            self._trocr_available = False
            return False
        
        try:
            from modules.line_segmentor import SSDLineSegmentor
            from modules.trocr_engine import TrOCREngine as TrOCR
            
            self._line_segmentor = SSDLineSegmentor(
                min_line_height=10,
                max_line_height=200,
                min_line_width=50,
                padding=5
            )
            
            self._trocr_engine = TrOCR()
            self._trocr_engine._ensure_initialized()
            
            print("TrOCR with line segmentation initialized successfully")
            self._trocr_available = True
            return True
            
        except Exception as e:
            print(f"TrOCR initialization failed: {e}")
            print("Falling back to Tesseract OCR")
            self._trocr_available = False
            return False
    
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract full text from image using TrOCR or Tesseract.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            Extracted text as string
        """
        # Try TrOCR first
        if self._init_trocr():
            try:
                text, _ = self._extract_text_trocr(image)
                return text
            except Exception as e:
                print(f"TrOCR extraction failed: {e}")
        
        # Fallback to Tesseract
        if self.use_tesseract_fallback:
            return self._extract_text_tesseract(image)
        
        return ""
    
    def _extract_text_tesseract(self, image: np.ndarray) -> str:
        """Extract text using Tesseract OCR."""
        text = pytesseract.image_to_string(image, lang=self.lang, config=self.config)
        return text.strip()
    
    def _extract_text_trocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text using TrOCR with line segmentation.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (extracted_text, average_confidence)
        """
        # Segment into lines
        line_segments = self._line_segmentor.segment_lines(image)
        
        if not line_segments:
            return "", 0.0
        
        # Extract just the images for batch processing
        line_images = [img for img, _ in line_segments]
        
        # Recognize all lines in batch
        results = self._trocr_engine.recognize_lines_batch(line_images)
        
        # Aggregate results
        texts = [r.text for r in results if r.text]
        confidences = [r.confidence for r in results]
        
        full_text = "\n".join(texts)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_confidence
    
    def extract_with_boxes(self, image: np.ndarray) -> List[TextBox]:
        """
        Extract text with bounding box positions using TrOCR or Tesseract.
        
        Args:
            image: Input image
            
        Returns:
            List of TextBox objects with position data
        """
        # Try TrOCR first
        if self._init_trocr():
            try:
                return self._extract_with_boxes_trocr(image)
            except Exception as e:
                print(f"TrOCR box extraction failed: {e}")
        
        # Fallback to Tesseract
        if self.use_tesseract_fallback:
            return self._extract_with_boxes_tesseract(image)
        
        return []
    
    def _extract_with_boxes_tesseract(self, image: np.ndarray) -> List[TextBox]:
        """Extract text with boxes using Tesseract."""
        data = pytesseract.image_to_data(
            image, lang=self.lang, config=self.config,
            output_type=pytesseract.Output.DICT
        )
        
        text_boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if not text or conf < 0:
                continue
            
            text_boxes.append(TextBox(
                text=text,
                x=data['left'][i],
                y=data['top'][i],
                width=data['width'][i],
                height=data['height'][i],
                confidence=conf / 100.0
            ))
        
        return text_boxes
    
    def _extract_with_boxes_trocr(self, image: np.ndarray) -> List[TextBox]:
        """
        Extract text with bounding boxes using TrOCR.
        
        Returns line-level text boxes (not word-level like Tesseract).
        """
        # Segment into lines
        line_segments = self._line_segmentor.segment_lines(image)
        
        if not line_segments:
            return []
        
        # Extract images and boxes
        line_images = [img for img, _ in line_segments]
        line_boxes = [box for _, box in line_segments]
        
        # Recognize all lines
        results = self._trocr_engine.recognize_lines_batch(line_images)
        
        # Create TextBox objects with line positions
        text_boxes = []
        for result, box in zip(results, line_boxes):
            if result.text:
                text_boxes.append(TextBox(
                    text=result.text,
                    x=box.x,
                    y=box.y,
                    width=box.width,
                    height=box.height,
                    confidence=result.confidence
                ))
        
        return text_boxes
    
    def detect_checkboxes(self, image: np.ndarray) -> List[Checkbox]:
        """
        Detect checkboxes in the image.
        
        Args:
            image: Input image (grayscale)
            
        Returns:
            List of detected Checkbox objects
        """
        checkboxes = []
        
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter for square-ish shapes (checkbox candidates)
            aspect_ratio = w / float(h) if h > 0 else 0
            
            # Checkbox criteria: roughly square, reasonable size
            if 0.7 < aspect_ratio < 1.3 and 15 < w < 50 and 15 < h < 50:
                # Check if it's filled (checked)
                roi = gray[y:y+h, x:x+w]
                
                # Calculate fill ratio (dark pixels vs total)
                _, binary_roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
                fill_ratio = np.sum(binary_roi == 255) / (w * h)
                
                is_checked = fill_ratio > 0.3  # >30% filled = checked
                
                checkboxes.append(Checkbox(
                    x=x, y=y, width=w, height=h,
                    is_checked=is_checked,
                    confidence=0.8 if 0.7 < aspect_ratio < 1.3 else 0.6
                ))
        
        return checkboxes
    
    def detect_signature(self, image: np.ndarray, region: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Detect if a signature is present in the image or region.
        
        Args:
            image: Input image
            region: Optional region to check (x, y, width, height)
            
        Returns:
            True if signature detected, False otherwise
        """
        # Extract region if specified
        if region:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # Ensure grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Apply edge detection for stroke analysis
        edges = cv2.Canny(gray, 50, 150)
        
        # Signatures typically have:
        # 1. Continuous strokes (connected components)
        # 2. Varying stroke widths
        # 3. Non-uniform patterns (unlike typed text)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return False
        
        # Analyze contour properties
        total_area = sum(cv2.contourArea(c) for c in contours)
        avg_contour_size = total_area / len(contours) if contours else 0
        
        # Signatures tend to have connected strokes (fewer, larger contours)
        # vs typed text (many small contours)
        has_signature_characteristics = (
            len(contours) < 100 and  # Not too fragmented
            avg_contour_size > 10 and  # Reasonably sized strokes
            total_area > 500  # Enough ink present
        )
        
        return has_signature_characteristics
    
    def process_image(self, image: np.ndarray) -> OCRResult:
        """
        Complete OCR processing of an image.
        
        Uses TrOCR with line segmentation for text extraction,
        with Tesseract as fallback.
        
        Args:
            image: Input image
            
        Returns:
            OCRResult with all extracted data
        """
        ocr_engine_used = "tesseract"
        
        # Try TrOCR first
        if self._init_trocr():
            try:
                # Use TrOCR for text extraction
                text_boxes = self._extract_with_boxes_trocr(image)
                full_text = "\n".join(tb.text for tb in text_boxes)
                ocr_engine_used = "trocr"
            except Exception as e:
                print(f"TrOCR processing failed: {e}")
                text_boxes = None
        else:
            text_boxes = None
        
        # Fallback to Tesseract
        if text_boxes is None and self.use_tesseract_fallback:
            text_boxes = self._extract_with_boxes_tesseract(image)
            full_text = self._extract_text_tesseract(image)
            ocr_engine_used = "tesseract"
        elif text_boxes is None:
            text_boxes = []
            full_text = ""
        
        # Detect checkboxes (always uses OpenCV)
        checkboxes = self.detect_checkboxes(image)
        
        # Detect signature (always uses OpenCV)
        has_signature = self.detect_signature(image)
        
        # Calculate average confidence
        if text_boxes:
            avg_conf = sum(tb.confidence for tb in text_boxes) / len(text_boxes)
        else:
            avg_conf = 0.0
        
        return OCRResult(
            full_text=full_text,
            text_boxes=text_boxes,
            checkboxes=checkboxes,
            has_signature=has_signature,
            average_confidence=avg_conf,
            ocr_engine_used=ocr_engine_used
        )
