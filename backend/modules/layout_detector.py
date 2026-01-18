"""
Layout Detection Module for Form Extraction.
Handles form structure recognition, key-value pair identification, and section classification.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .ocr_engine import TextBox


@dataclass
class KeyValuePair:
    """Represents a detected key-value pair from a form."""
    key: str
    value: str
    key_bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    value_bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    section: Optional[str] = None


@dataclass
class FormRegion:
    """Represents a detected region/section of a form."""
    name: str
    x: int
    y: int
    width: int
    height: int
    fields: List[KeyValuePair]


class LayoutDetector:
    """Detects form layout structure and extracts key-value pairs."""
    
    # Keywords for section classification (Banking form specific)
    SECTION_KEYWORDS = {
        'personal_info': ['name', 'full name', 'first name', 'last name', 'dob', 'date of birth', 
                         'gender', 'age', 'nationality', 'marital status', 'father'],
        'address': ['address', 'city', 'state', 'pin', 'pincode', 'zip', 'country', 'district',
                   'permanent address', 'current address', 'correspondence'],
        'contact': ['phone', 'mobile', 'email', 'telephone', 'contact', 'fax'],
        'identity': ['pan', 'aadhaar', 'aadhar', 'passport', 'voter id', 'driving license',
                    'id proof', 'identity'],
        'employment': ['employer', 'company', 'occupation', 'designation', 'salary', 'income',
                      'employment', 'organization', 'business'],
        'banking': ['account', 'ifsc', 'branch', 'bank', 'nominee', 'account type',
                   'savings', 'current', 'deposit']
    }
    
    # Common label endings indicating a field label
    LABEL_INDICATORS = [':', '-', '.']
    
    def __init__(self, proximity_threshold: int = 50):
        """
        Initialize Layout Detector.
        
        Args:
            proximity_threshold: Maximum distance between label and value to be paired
        """
        self.proximity_threshold = proximity_threshold
    
    def detect_key_value_pairs(self, text_boxes: List[TextBox]) -> List[KeyValuePair]:
        """
        Identify key-value pairs from OCR text boxes.
        
        Args:
            text_boxes: List of TextBox objects from OCR
            
        Returns:
            List of KeyValuePair objects
        """
        pairs = []
        used_indices = set()
        
        for i, box in enumerate(text_boxes):
            if i in used_indices:
                continue
            
            text = box.text.strip()
            
            # Check if this looks like a label
            if self._is_label(text):
                # Clean the label
                label = self._clean_label(text)
                
                # Find the nearest value (to the right or below)
                value_box, value_idx = self._find_nearest_value(
                    box, text_boxes, used_indices
                )
                
                if value_box:
                    used_indices.add(i)
                    used_indices.add(value_idx)
                    
                    # Classify section
                    section = self._classify_section(label)
                    
                    pairs.append(KeyValuePair(
                        key=label,
                        value=value_box.text.strip(),
                        key_bbox=box.bbox,
                        value_bbox=value_box.bbox,
                        confidence=(box.confidence + value_box.confidence) / 2,
                        section=section
                    ))
        
        return pairs
    
    def _is_label(self, text: str) -> bool:
        """Check if text appears to be a field label."""
        # Check for common label endings
        for indicator in self.LABEL_INDICATORS:
            if text.endswith(indicator):
                return True
        
        # Check if it matches known field names
        text_lower = text.lower()
        for section_keywords in self.SECTION_KEYWORDS.values():
            for keyword in section_keywords:
                if keyword in text_lower:
                    return True
        
        return False
    
    def _clean_label(self, text: str) -> str:
        """Remove label indicators and clean up the label text."""
        cleaned = text.strip()
        for indicator in self.LABEL_INDICATORS:
            if cleaned.endswith(indicator):
                cleaned = cleaned[:-1].strip()
        return cleaned
    
    def _find_nearest_value(
        self, 
        label_box: TextBox, 
        text_boxes: List[TextBox],
        used_indices: set
    ) -> Tuple[Optional[TextBox], Optional[int]]:
        """
        Find the nearest text box that could be the value for this label.
        Prioritizes boxes to the right, then below.
        
        Args:
            label_box: The label TextBox
            text_boxes: All text boxes
            used_indices: Indices already used
            
        Returns:
            Tuple of (value TextBox, index) or (None, None)
        """
        label_center = label_box.center
        label_right = label_box.x + label_box.width
        label_bottom = label_box.y + label_box.height
        
        best_right = None
        best_right_dist = float('inf')
        best_right_idx = None
        
        best_below = None
        best_below_dist = float('inf')
        best_below_idx = None
        
        for i, box in enumerate(text_boxes):
            if i in used_indices or box == label_box:
                continue
            
            # Skip if it looks like another label
            if self._is_label(box.text):
                continue
            
            box_center = box.center
            
            # Check if box is to the right (same horizontal line)
            if (abs(box.y - label_box.y) < label_box.height * 0.8 and 
                box.x > label_right):
                dist = box.x - label_right
                if dist < self.proximity_threshold and dist < best_right_dist:
                    best_right = box
                    best_right_dist = dist
                    best_right_idx = i
            
            # Check if box is below (same vertical column)
            if (abs(box.x - label_box.x) < label_box.width * 0.5 and 
                box.y > label_bottom):
                dist = box.y - label_bottom
                if dist < self.proximity_threshold and dist < best_below_dist:
                    best_below = box
                    best_below_dist = dist
                    best_below_idx = i
        
        # Prefer right-side values over below values
        if best_right:
            return best_right, best_right_idx
        elif best_below:
            return best_below, best_below_idx
        
        return None, None
    
    def _classify_section(self, label: str) -> Optional[str]:
        """
        Classify which section a label belongs to.
        
        Args:
            label: The field label text
            
        Returns:
            Section name or None
        """
        label_lower = label.lower()
        
        for section, keywords in self.SECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in label_lower:
                    return section
        
        return None
    
    def detect_form_regions(self, image: np.ndarray) -> List[FormRegion]:
        """
        Detect distinct regions/sections in a form using contour analysis.
        
        Args:
            image: Input image
            
        Returns:
            List of FormRegion objects
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect horizontal and vertical lines (form structure)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect lines
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine lines
        lines = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours of form sections
        contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w > 100 and h > 50:
                regions.append(FormRegion(
                    name=f"region_{i}",
                    x=x, y=y, width=w, height=h,
                    fields=[]
                ))
        
        return regions
    
    def group_pairs_by_section(self, pairs: List[KeyValuePair]) -> Dict[str, List[KeyValuePair]]:
        """
        Group key-value pairs by their section.
        
        Args:
            pairs: List of KeyValuePair objects
            
        Returns:
            Dictionary mapping section names to pairs
        """
        grouped = {}
        
        for pair in pairs:
            section = pair.section or 'other'
            if section not in grouped:
                grouped[section] = []
            grouped[section].append(pair)
        
        return grouped
    
    def visualize_layout(
        self, 
        image: np.ndarray, 
        pairs: List[KeyValuePair],
        show_boxes: bool = True
    ) -> np.ndarray:
        """
        Draw detected key-value pairs on the image.
        
        Args:
            image: Input image
            pairs: Detected key-value pairs
            show_boxes: Whether to draw bounding boxes
            
        Returns:
            Image with visualizations
        """
        result = image.copy()
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        # Color scheme for different sections
        colors = {
            'personal_info': (0, 255, 0),    # Green
            'address': (255, 165, 0),         # Orange
            'contact': (0, 165, 255),         # Light blue
            'identity': (255, 0, 255),        # Magenta
            'employment': (255, 255, 0),      # Cyan
            'banking': (0, 255, 255),         # Yellow
            'other': (128, 128, 128)          # Gray
        }
        
        for pair in pairs:
            section = pair.section or 'other'
            color = colors.get(section, (128, 128, 128))
            
            if show_boxes and pair.key_bbox:
                x, y, w, h = pair.key_bbox
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result, pair.key, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if show_boxes and pair.value_bbox:
                x, y, w, h = pair.value_bbox
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 1)
        
        return result
