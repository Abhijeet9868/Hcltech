"""
TrOCR Engine Module for Transformer-based OCR.
Uses Microsoft TrOCR for high-accuracy text recognition on line images.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

# Lazy loading for heavy dependencies
_trocr_processor = None
_trocr_model = None
_torch = None
_device = None


def _get_device():
    """Get the best available device (CUDA or CPU)."""
    global _torch, _device
    
    if _device is not None:
        return _device
    
    import torch
    _torch = torch
    
    device_config = getattr(Config, 'TROCR_DEVICE', 'auto')
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            _device = torch.device('cuda')
            print("TrOCR: Using CUDA GPU")
        else:
            _device = torch.device('cpu')
            print("TrOCR: Using CPU")
    elif device_config == 'cuda' and torch.cuda.is_available():
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')
    
    return _device


def _load_trocr_model(model_name: str = None):
    """Lazy load TrOCR model and processor."""
    global _trocr_processor, _trocr_model
    
    if _trocr_model is not None:
        return _trocr_processor, _trocr_model
    
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    
    if model_name is None:
        model_name = getattr(Config, 'TROCR_MODEL', 'microsoft/trocr-base-handwritten')
    
    print(f"Loading TrOCR model: {model_name}")
    
    _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
    _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    device = _get_device()
    _trocr_model = _trocr_model.to(device)
    _trocr_model.eval()
    
    print(f"TrOCR model loaded successfully on {device}")
    
    return _trocr_processor, _trocr_model


@dataclass
class TrOCRResult:
    """Result from TrOCR text recognition."""
    text: str
    confidence: float
    line_index: int = 0
    
    def __str__(self):
        return self.text


class TrOCREngine:
    """
    Transformer-based OCR Engine using Microsoft TrOCR.
    
    TrOCR is a vision-encoder-decoder model that achieves state-of-the-art
    results on text recognition, especially for handwritten text.
    """
    
    # Available TrOCR models
    MODELS = {
        'small-printed': 'microsoft/trocr-small-printed',
        'base-printed': 'microsoft/trocr-base-printed',
        'small-handwritten': 'microsoft/trocr-small-handwritten',
        'base-handwritten': 'microsoft/trocr-base-handwritten',
        'large-handwritten': 'microsoft/trocr-large-handwritten',
    }
    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        batch_size: int = None,
        max_length: int = 128
    ):
        """
        Initialize TrOCR Engine.
        
        Args:
            model_name: HuggingFace model name or shorthand (e.g., 'base-handwritten')
            device: Device to use ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for processing multiple lines
            max_length: Maximum sequence length for generation
        """
        # Resolve model name
        if model_name is None:
            model_name = getattr(Config, 'TROCR_MODEL', 'microsoft/trocr-base-handwritten')
        elif model_name in self.MODELS:
            model_name = self.MODELS[model_name]
        
        self.model_name = model_name
        self.batch_size = batch_size or getattr(Config, 'TROCR_BATCH_SIZE', 8)
        self.max_length = max_length
        
        # Model will be loaded lazily on first use
        self._processor = None
        self._model = None
        self._device = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure model is loaded before use."""
        if not self._initialized:
            self._processor, self._model = _load_trocr_model(self.model_name)
            self._device = _get_device()
            self._initialized = True
    
    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """
        Prepare image for TrOCR processing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            PIL Image in RGB format
        """
        import cv2
        
        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 3:  # BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(image)
    
    def recognize_line(self, line_image: np.ndarray) -> TrOCRResult:
        """
        Recognize text from a single line image.
        
        Args:
            line_image: Image containing a single line of text
            
        Returns:
            TrOCRResult with recognized text and confidence
        """
        self._ensure_initialized()
        
        # Prepare image
        pil_image = self._prepare_image(line_image)
        
        # Process with TrOCR
        pixel_values = self._processor(
            images=pil_image,
            return_tensors="pt"
        ).pixel_values.to(self._device)
        
        # Generate text
        with _torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode text
        generated_ids = outputs.sequences
        generated_text = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Calculate confidence from scores
        confidence = self._calculate_confidence(outputs)
        
        return TrOCRResult(
            text=generated_text.strip(),
            confidence=confidence
        )
    
    def recognize_lines_batch(
        self,
        line_images: List[np.ndarray],
        batch_size: int = None
    ) -> List[TrOCRResult]:
        """
        Recognize text from multiple line images using batching.
        
        Args:
            line_images: List of line images
            batch_size: Override default batch size
            
        Returns:
            List of TrOCRResult objects
        """
        self._ensure_initialized()
        
        if not line_images:
            return []
        
        batch_size = batch_size or self.batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(line_images), batch_size):
            batch = line_images[i:i + batch_size]
            batch_results = self._process_batch(batch, start_index=i)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(
        self,
        images: List[np.ndarray],
        start_index: int = 0
    ) -> List[TrOCRResult]:
        """
        Process a batch of images.
        
        Args:
            images: Batch of line images
            start_index: Starting index for line numbering
            
        Returns:
            List of TrOCRResult objects
        """
        # Prepare all images
        pil_images = [self._prepare_image(img) for img in images]
        
        # Process batch
        pixel_values = self._processor(
            images=pil_images,
            return_tensors="pt",
            padding=True
        ).pixel_values.to(self._device)
        
        # Generate text for batch
        with _torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                max_length=self.max_length,
                num_beams=4,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode all texts
        generated_ids = outputs.sequences
        generated_texts = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        
        # Calculate per-sequence confidence
        confidences = self._calculate_batch_confidence(outputs, len(images))
        
        results = []
        for idx, (text, conf) in enumerate(zip(generated_texts, confidences)):
            results.append(TrOCRResult(
                text=text.strip(),
                confidence=conf,
                line_index=start_index + idx
            ))
        
        return results
    
    def _calculate_confidence(self, outputs) -> float:
        """
        Calculate confidence score from generation outputs.
        
        Uses the average probability of generated tokens.
        """
        if not hasattr(outputs, 'scores') or outputs.scores is None:
            return 0.8  # Default confidence if scores unavailable
        
        try:
            import torch.nn.functional as F
            
            # Get probabilities from scores
            scores = outputs.scores
            if not scores:
                return 0.8
            
            # Calculate average probability across all generated tokens
            probs = []
            for score in scores:
                prob = F.softmax(score, dim=-1)
                max_prob = prob.max(dim=-1).values.item()
                probs.append(max_prob)
            
            return sum(probs) / len(probs) if probs else 0.8
            
        except Exception:
            return 0.8
    
    def _calculate_batch_confidence(
        self,
        outputs,
        batch_size: int
    ) -> List[float]:
        """Calculate confidence scores for a batch."""
        if not hasattr(outputs, 'scores') or outputs.scores is None:
            return [0.8] * batch_size
        
        try:
            import torch.nn.functional as F
            
            scores = outputs.scores
            if not scores:
                return [0.8] * batch_size
            
            # Calculate per-sequence confidence
            confidences = []
            for batch_idx in range(batch_size):
                probs = []
                for score in scores:
                    if batch_idx < score.shape[0]:
                        prob = F.softmax(score[batch_idx:batch_idx+1], dim=-1)
                        max_prob = prob.max(dim=-1).values.item()
                        probs.append(max_prob)
                
                conf = sum(probs) / len(probs) if probs else 0.8
                confidences.append(conf)
            
            return confidences
            
        except Exception:
            return [0.8] * batch_size
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        self._ensure_initialized()
        
        return {
            'model_name': self.model_name,
            'device': str(self._device),
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'num_parameters': sum(
                p.numel() for p in self._model.parameters()
            ) if self._model else 0
        }
    
    def recognize_document(
        self,
        line_images: List[np.ndarray],
        separator: str = "\n"
    ) -> Tuple[str, float]:
        """
        Recognize text from a full document (list of line images).
        
        Args:
            line_images: List of line images in reading order
            separator: String to join lines with
            
        Returns:
            Tuple of (full_text, average_confidence)
        """
        results = self.recognize_lines_batch(line_images)
        
        if not results:
            return "", 0.0
        
        full_text = separator.join(r.text for r in results if r.text)
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return full_text, avg_confidence


class TrOCREngineWithFallback:
    """
    TrOCR Engine with Tesseract fallback for robustness.
    Falls back to Tesseract if TrOCR fails or is unavailable.
    """
    
    def __init__(self, use_tesseract_fallback: bool = True):
        """
        Initialize with optional Tesseract fallback.
        
        Args:
            use_tesseract_fallback: Whether to use Tesseract as fallback
        """
        self.use_tesseract_fallback = use_tesseract_fallback
        self._trocr = None
        self._trocr_available = None
    
    def _init_trocr(self) -> bool:
        """Try to initialize TrOCR."""
        if self._trocr_available is not None:
            return self._trocr_available
        
        try:
            self._trocr = TrOCREngine()
            self._trocr._ensure_initialized()
            self._trocr_available = True
            return True
        except Exception as e:
            print(f"TrOCR initialization failed: {e}")
            self._trocr_available = False
            return False
    
    def recognize_line(self, line_image: np.ndarray) -> TrOCRResult:
        """Recognize text with TrOCR or fallback to Tesseract."""
        if self._init_trocr():
            try:
                return self._trocr.recognize_line(line_image)
            except Exception as e:
                print(f"TrOCR recognition failed: {e}")
        
        # Fallback to Tesseract
        if self.use_tesseract_fallback:
            return self._tesseract_fallback(line_image)
        
        return TrOCRResult(text="", confidence=0.0)
    
    def _tesseract_fallback(self, line_image: np.ndarray) -> TrOCRResult:
        """Use Tesseract as fallback."""
        import pytesseract
        
        try:
            text = pytesseract.image_to_string(
                line_image, config='--psm 7'  # Single line mode
            ).strip()
            
            # Get confidence
            data = pytesseract.image_to_data(
                line_image,
                config='--psm 7',
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [
                float(c) for c in data['conf'] if float(c) > 0
            ]
            avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.5
            
            return TrOCRResult(text=text, confidence=avg_conf)
            
        except Exception as e:
            print(f"Tesseract fallback failed: {e}")
            return TrOCRResult(text="", confidence=0.0)
    
    def recognize_lines_batch(
        self,
        line_images: List[np.ndarray]
    ) -> List[TrOCRResult]:
        """Batch recognize with fallback."""
        if self._init_trocr():
            try:
                return self._trocr.recognize_lines_batch(line_images)
            except Exception as e:
                print(f"TrOCR batch recognition failed: {e}")
        
        # Fallback to individual Tesseract calls
        if self.use_tesseract_fallback:
            return [self._tesseract_fallback(img) for img in line_images]
        
        return [TrOCRResult(text="", confidence=0.0) for _ in line_images]
