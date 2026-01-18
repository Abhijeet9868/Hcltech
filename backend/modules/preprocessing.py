"""
Image Preprocessing Module for Form Extraction.
Handles deskewing, denoising, and image enhancement for OCR.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, List, Optional
import io


class ImagePreprocessor:
    """Preprocesses images for optimal OCR performance."""
    
    def __init__(self, target_dpi: int = 300):
        self.target_dpi = target_dpi
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline.
        
        Args:
            image: Input image as numpy array (BGR format from cv2)
            
        Returns:
            Preprocessed image ready for OCR
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply preprocessing pipeline
        gray = self.deskew(gray)
        gray = self.denoise(gray)
        gray = self.enhance_contrast(gray)
        gray = self.binarize(gray)
        
        return gray
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct image rotation/skew using Hough transform.
        
        Args:
            image: Grayscale image
            
        Returns:
            Deskewed image
        """
        # Detect edges
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None:
            return image
        
        # Calculate average angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # Only consider near-horizontal lines
                angles.append(angle)
        
        if not angles:
            return image
        
        # Get median angle to ignore outliers
        median_angle = np.median(angles)
        
        # Rotate image to correct skew
        if abs(median_angle) > 0.5:  # Only rotate if skew is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, rotation_matrix, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise from image using bilateral filtering.
        
        Args:
            image: Grayscale image
            
        Returns:
            Denoised image
        """
        # Bilateral filter preserves edges while removing noise
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        return enhanced
    
    def binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Convert to binary image using adaptive thresholding.
        
        Args:
            image: Grayscale image
            
        Returns:
            Binary image
        """
        binary = cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return binary
    
    def resize_for_ocr(self, image: np.ndarray, scale_factor: float = 2.0) -> np.ndarray:
        """
        Upscale image for better OCR accuracy.
        
        Args:
            image: Input image
            scale_factor: Factor to scale image by
            
        Returns:
            Resized image
        """
        width = int(image.shape[1] * scale_factor)
        height = int(image.shape[0] * scale_factor)
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        return resized
    
    def remove_borders(self, image: np.ndarray, border_size: int = 10) -> np.ndarray:
        """
        Remove dark borders from scanned images.
        
        Args:
            image: Input image
            border_size: Size of border to remove
            
        Returns:
            Image with borders removed
        """
        h, w = image.shape[:2]
        return image[border_size:h-border_size, border_size:w-border_size]
    
    def load_image(self, file_path: str) -> np.ndarray:
        """
        Load image from file path.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Image as numpy array
        """
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError(f"Could not load image from {file_path}")
        return image
    
    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Load image from bytes.
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Image as numpy array
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
        return image
