"""
Line Segmentation Module using SSD-based / Morphological Detection.
Segments document images into individual text lines for TrOCR processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


@dataclass
class LineBox:
    """Represents a detected text line region."""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    @property
    def center_y(self) -> int:
        """Return vertical center of the line."""
        return self.y + self.height // 2
    
    @property
    def area(self) -> int:
        """Return area of the bounding box."""
        return self.width * self.height


class SSDLineSegmentor:
    """
    Text line segmentation using morphological operations and contour detection.
    
    This is a hybrid approach that can be enhanced with SSD deep learning model
    for more accurate line detection when available.
    """
    
    def __init__(
        self,
        min_line_height: int = 10,
        max_line_height: int = 200,
        min_line_width: int = 50,
        confidence_threshold: float = 0.5,
        padding: int = 5,
        use_ssd: bool = False,
        ssd_model_path: Optional[str] = None
    ):
        """
        Initialize the line segmentor.
        
        Args:
            min_line_height: Minimum height for a valid text line
            max_line_height: Maximum height for a valid text line
            min_line_width: Minimum width for a valid text line
            confidence_threshold: Confidence threshold for SSD detection
            padding: Padding to add around detected lines
            use_ssd: Whether to use SSD model (if available)
            ssd_model_path: Path to custom SSD model
        """
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_line_width = min_line_width
        self.confidence_threshold = confidence_threshold
        self.padding = padding
        self.use_ssd = use_ssd
        self.ssd_model = None
        
        # Try to load SSD model if requested
        if use_ssd and ssd_model_path:
            self._load_ssd_model(ssd_model_path)
    
    def _load_ssd_model(self, model_path: str) -> bool:
        """
        Load SSD model for line detection.
        
        Args:
            model_path: Path to the SSD model
            
        Returns:
            True if model loaded successfully
        """
        try:
            # Load SSD model using OpenCV DNN
            self.ssd_model = cv2.dnn.readNetFromCaffe(
                model_path + '.prototxt',
                model_path + '.caffemodel'
            )
            return True
        except Exception as e:
            print(f"Warning: Could not load SSD model: {e}")
            print("Falling back to morphological detection")
            self.ssd_model = None
            return False
    
    def detect_lines(self, image: np.ndarray) -> List[LineBox]:
        """
        Detect text lines in the image.
        
        Uses SSD model if available, otherwise falls back to
        morphological operations and contour detection.
        
        Args:
            image: Input image (grayscale or color)
            
        Returns:
            List of LineBox objects representing detected text lines
        """
        if self.ssd_model is not None:
            return self._detect_lines_ssd(image)
        else:
            return self._detect_lines_morphological(image)
    
    def _detect_lines_ssd(self, image: np.ndarray) -> List[LineBox]:
        """
        Detect lines using SSD deep learning model.
        
        Args:
            image: Input image
            
        Returns:
            List of detected LineBox objects
        """
        # Prepare image for SSD
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=True, crop=False
        )
        
        self.ssd_model.setInput(blob)
        detections = self.ssd_model.forward()
        
        lines = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                
                lines.append(LineBox(
                    x=max(0, x1),
                    y=max(0, y1),
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=float(confidence)
                ))
        
        return self.sort_lines_reading_order(lines)
    
    def _detect_lines_morphological(self, image: np.ndarray) -> List[LineBox]:
        """
        Detect lines using morphological operations (fallback method).
        
        This method uses dilation with horizontal kernels to connect
        text characters into lines, then finds contours.
        
        Args:
            image: Input image
            
        Returns:
            List of detected LineBox objects
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize the image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create horizontal kernel to connect text in lines
        # Kernel width should be proportional to image width
        kernel_width = max(image.shape[1] // 20, 30)
        kernel_height = 1
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_width, kernel_height)
        )
        
        # Dilate to connect text horizontally
        dilated = cv2.dilate(binary, horizontal_kernel, iterations=2)
        
        # Apply vertical dilation to merge close lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(dilated, vertical_kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        lines = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size constraints
            if (self.min_line_height <= h <= self.max_line_height and
                w >= self.min_line_width):
                
                # Calculate confidence based on aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0
                # Text lines typically have aspect ratio > 3
                confidence = min(1.0, aspect_ratio / 10.0) if aspect_ratio > 1 else 0.5
                
                lines.append(LineBox(
                    x=x, y=y, width=w, height=h,
                    confidence=confidence
                ))
        
        return self.sort_lines_reading_order(lines)
    
    def sort_lines_reading_order(self, lines: List[LineBox]) -> List[LineBox]:
        """
        Sort detected lines in reading order (top-to-bottom, left-to-right).
        
        Args:
            lines: List of LineBox objects
            
        Returns:
            Sorted list of LineBox objects
        """
        if not lines:
            return lines
        
        # Group lines by vertical position (with tolerance)
        line_groups = []
        tolerance = 20  # pixels
        
        sorted_by_y = sorted(lines, key=lambda l: l.y)
        
        current_group = [sorted_by_y[0]]
        current_y = sorted_by_y[0].center_y
        
        for line in sorted_by_y[1:]:
            if abs(line.center_y - current_y) <= tolerance:
                current_group.append(line)
            else:
                line_groups.append(sorted(current_group, key=lambda l: l.x))
                current_group = [line]
                current_y = line.center_y
        
        line_groups.append(sorted(current_group, key=lambda l: l.x))
        
        # Flatten the groups
        sorted_lines = []
        for group in line_groups:
            sorted_lines.extend(group)
        
        return sorted_lines
    
    def segment_lines(
        self,
        image: np.ndarray,
        line_boxes: Optional[List[LineBox]] = None
    ) -> List[Tuple[np.ndarray, LineBox]]:
        """
        Segment the image into individual line images.
        
        Args:
            image: Input image
            line_boxes: Optional pre-detected line boxes
            
        Returns:
            List of tuples (line_image, line_box)
        """
        if line_boxes is None:
            line_boxes = self.detect_lines(image)
        
        h, w = image.shape[:2]
        line_images = []
        
        for box in line_boxes:
            # Add padding around the line
            x1 = max(0, box.x - self.padding)
            y1 = max(0, box.y - self.padding)
            x2 = min(w, box.x + box.width + self.padding)
            y2 = min(h, box.y + box.height + self.padding)
            
            # Crop the line
            line_img = image[y1:y2, x1:x2]
            
            if line_img.size > 0:
                line_images.append((line_img, box))
        
        return line_images
    
    def merge_overlapping_lines(
        self,
        lines: List[LineBox],
        overlap_threshold: float = 0.5
    ) -> List[LineBox]:
        """
        Merge overlapping line detections.
        
        Args:
            lines: List of LineBox objects
            overlap_threshold: IoU threshold for merging
            
        Returns:
            Merged list of LineBox objects
        """
        if not lines:
            return lines
        
        # Sort by area (largest first)
        lines = sorted(lines, key=lambda l: l.area, reverse=True)
        
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            # Find overlapping lines
            to_merge = [line1]
            
            for j, line2 in enumerate(lines[i+1:], i+1):
                if j in used:
                    continue
                
                iou = self._calculate_iou(line1, line2)
                if iou > overlap_threshold:
                    to_merge.append(line2)
                    used.add(j)
            
            # Merge the lines
            if len(to_merge) > 1:
                merged_line = self._merge_boxes(to_merge)
                merged.append(merged_line)
            else:
                merged.append(line1)
            
            used.add(i)
        
        return merged
    
    def _calculate_iou(self, box1: LineBox, box2: LineBox) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.width, box2.x + box2.width)
        y2 = min(box1.y + box1.height, box2.y + box2.height)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = box1.area + box2.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_boxes(self, boxes: List[LineBox]) -> LineBox:
        """Merge multiple boxes into one bounding box."""
        x1 = min(b.x for b in boxes)
        y1 = min(b.y for b in boxes)
        x2 = max(b.x + b.width for b in boxes)
        y2 = max(b.y + b.height for b in boxes)
        
        avg_confidence = sum(b.confidence for b in boxes) / len(boxes)
        
        return LineBox(
            x=x1, y=y1,
            width=x2 - x1,
            height=y2 - y1,
            confidence=avg_confidence
        )
    
    def visualize_lines(
        self,
        image: np.ndarray,
        lines: List[LineBox],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detected lines on the image for visualization.
        
        Args:
            image: Input image
            lines: List of LineBox objects
            color: BGR color for the rectangles
            thickness: Line thickness
            
        Returns:
            Image with drawn rectangles
        """
        vis_image = image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        for i, line in enumerate(lines):
            cv2.rectangle(
                vis_image,
                (line.x, line.y),
                (line.x + line.width, line.y + line.height),
                color, thickness
            )
            # Add line number
            cv2.putText(
                vis_image, str(i + 1),
                (line.x, line.y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1
            )
        
        return vis_image
