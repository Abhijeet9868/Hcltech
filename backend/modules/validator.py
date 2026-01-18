"""
Validator Module for Form Extraction.
Handles field validation, normalization, and confidence scoring.
"""

import re
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of field validation."""
    is_valid: bool
    normalized_value: Any
    confidence: float
    needs_review: bool
    errors: List[str]


class FieldValidator:
    """Validates and normalizes extracted form fields."""
    
    # Validation patterns
    PATTERNS = {
        'email': r'^[\w\.-]+@[\w\.-]+\.\w{2,}$',
        'phone': r'^[\+]?[\d\s\-\(\)]{10,15}$',
        'pan': r'^[A-Z]{5}[0-9]{4}[A-Z]$',
        'aadhaar': r'^\d{12}$',
        'pincode': r'^\d{6}$',
        'ifsc': r'^[A-Z]{4}0[A-Z0-9]{6}$',
    }
    
    # Date format patterns to try
    DATE_FORMATS = [
        '%Y-%m-%d',      # 2024-01-15
        '%d-%m-%Y',      # 15-01-2024
        '%d/%m/%Y',      # 15/01/2024
        '%m/%d/%Y',      # 01/15/2024
        '%d.%m.%Y',      # 15.01.2024
        '%B %d, %Y',     # January 15, 2024
        '%d %B %Y',      # 15 January 2024
        '%d-%b-%Y',      # 15-Jan-2024
    ]
    
    def __init__(self, high_confidence_threshold: float = 0.8, low_confidence_threshold: float = 0.6):
        """
        Initialize validator.
        
        Args:
            high_confidence_threshold: Threshold for high confidence
            low_confidence_threshold: Threshold below which review is needed
        """
        self.high_threshold = high_confidence_threshold
        self.low_threshold = low_confidence_threshold
    
    def validate_field(
        self, 
        field_name: str, 
        value: Any, 
        ocr_confidence: float = 1.0
    ) -> ValidationResult:
        """
        Validate and normalize a single field.
        
        Args:
            field_name: Name of the field
            value: Extracted value
            ocr_confidence: OCR confidence score
            
        Returns:
            ValidationResult with normalized value and status
        """
        if value is None or str(value).strip() == '':
            return ValidationResult(
                is_valid=False,
                normalized_value=None,
                confidence=0.0,
                needs_review=True,
                errors=['Field is empty']
            )
        
        value_str = str(value).strip()
        errors = []
        
        # Field-specific validation
        if 'email' in field_name.lower():
            result = self._validate_email(value_str, ocr_confidence)
        elif 'phone' in field_name.lower() or 'mobile' in field_name.lower():
            result = self._validate_phone(value_str, ocr_confidence)
        elif 'pan' in field_name.lower():
            result = self._validate_pan(value_str, ocr_confidence)
        elif 'aadhaar' in field_name.lower() or 'aadhar' in field_name.lower():
            result = self._validate_aadhaar(value_str, ocr_confidence)
        elif 'pincode' in field_name.lower() or 'pin' in field_name.lower() or 'zip' in field_name.lower():
            result = self._validate_pincode(value_str, ocr_confidence)
        elif 'date' in field_name.lower() or 'dob' in field_name.lower() or 'birth' in field_name.lower():
            result = self._validate_date(value_str, ocr_confidence)
        elif 'name' in field_name.lower():
            result = self._validate_name(value_str, ocr_confidence)
        elif 'ifsc' in field_name.lower():
            result = self._validate_ifsc(value_str, ocr_confidence)
        else:
            # Generic validation
            result = self._validate_generic(value_str, ocr_confidence)
        
        return result
    
    def _validate_email(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate email address."""
        value_clean = value.lower().strip()
        is_valid = bool(re.match(self.PATTERNS['email'], value_clean))
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=value_clean if is_valid else value,
            confidence=self._calculate_confidence(ocr_conf, is_valid),
            needs_review=not is_valid or ocr_conf < self.low_threshold,
            errors=[] if is_valid else ['Invalid email format']
        )
    
    def _validate_phone(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate phone number."""
        # Clean phone number
        digits_only = re.sub(r'[^\d]', '', value)
        
        # Check if it has valid digit count
        is_valid = 10 <= len(digits_only) <= 12
        
        # Format normalized value
        if len(digits_only) == 10:
            normalized = digits_only
        elif len(digits_only) == 12 and digits_only.startswith('91'):
            normalized = '+91 ' + digits_only[2:]
        else:
            normalized = digits_only
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=normalized,
            confidence=self._calculate_confidence(ocr_conf, is_valid),
            needs_review=not is_valid or ocr_conf < self.low_threshold,
            errors=[] if is_valid else ['Invalid phone number format']
        )
    
    def _validate_pan(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate PAN card number."""
        value_clean = value.upper().strip().replace(' ', '')
        is_valid = bool(re.match(self.PATTERNS['pan'], value_clean))
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=value_clean if is_valid else value,
            confidence=self._calculate_confidence(ocr_conf, is_valid),
            needs_review=not is_valid or ocr_conf < self.low_threshold,
            errors=[] if is_valid else ['Invalid PAN format (expected: ABCDE1234F)']
        )
    
    def _validate_aadhaar(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate Aadhaar number."""
        digits_only = re.sub(r'[^\d]', '', value)
        is_valid = len(digits_only) == 12
        
        # Format as XXXX XXXX XXXX
        if is_valid:
            normalized = f"{digits_only[:4]} {digits_only[4:8]} {digits_only[8:]}"
        else:
            normalized = value
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=normalized,
            confidence=self._calculate_confidence(ocr_conf, is_valid),
            needs_review=not is_valid or ocr_conf < self.low_threshold,
            errors=[] if is_valid else ['Invalid Aadhaar format (expected: 12 digits)']
        )
    
    def _validate_pincode(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate PIN code."""
        digits_only = re.sub(r'[^\d]', '', value)
        is_valid = len(digits_only) == 6
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=digits_only if is_valid else value,
            confidence=self._calculate_confidence(ocr_conf, is_valid),
            needs_review=not is_valid or ocr_conf < self.low_threshold,
            errors=[] if is_valid else ['Invalid PIN code (expected: 6 digits)']
        )
    
    def _validate_date(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate and normalize date."""
        parsed_date = None
        
        for fmt in self.DATE_FORMATS:
            try:
                parsed_date = datetime.strptime(value.strip(), fmt).date()
                break
            except ValueError:
                continue
        
        if parsed_date:
            # Validate date is reasonable (not in future for DOB, not too old)
            is_valid = True
            errors = []
            
            if parsed_date > date.today():
                is_valid = False
                errors.append('Date is in the future')
            elif parsed_date.year < 1900:
                is_valid = False
                errors.append('Date is too old')
            
            return ValidationResult(
                is_valid=is_valid,
                normalized_value=parsed_date.isoformat(),  # YYYY-MM-DD format
                confidence=self._calculate_confidence(ocr_conf, is_valid),
                needs_review=not is_valid or ocr_conf < self.low_threshold,
                errors=errors
            )
        
        return ValidationResult(
            is_valid=False,
            normalized_value=value,
            confidence=ocr_conf * 0.5,
            needs_review=True,
            errors=['Could not parse date format']
        )
    
    def _validate_name(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate name field."""
        # Names should only contain letters, spaces, dots, and common punctuation
        is_valid = bool(re.match(r'^[A-Za-z\s\.\'\-]+$', value))
        
        # Normalize: proper case
        normalized = ' '.join(word.capitalize() for word in value.split())
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=normalized,
            confidence=self._calculate_confidence(ocr_conf, is_valid),
            needs_review=not is_valid or ocr_conf < self.low_threshold,
            errors=[] if is_valid else ['Name contains invalid characters']
        )
    
    def _validate_ifsc(self, value: str, ocr_conf: float) -> ValidationResult:
        """Validate IFSC code."""
        value_clean = value.upper().strip().replace(' ', '')
        is_valid = bool(re.match(self.PATTERNS['ifsc'], value_clean))
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=value_clean if is_valid else value,
            confidence=self._calculate_confidence(ocr_conf, is_valid),
            needs_review=not is_valid or ocr_conf < self.low_threshold,
            errors=[] if is_valid else ['Invalid IFSC format']
        )
    
    def _validate_generic(self, value: str, ocr_conf: float) -> ValidationResult:
        """Generic validation for unspecified fields."""
        # Just clean whitespace
        normalized = ' '.join(value.split())
        is_valid = len(normalized) > 0
        
        return ValidationResult(
            is_valid=is_valid,
            normalized_value=normalized,
            confidence=ocr_conf,
            needs_review=ocr_conf < self.low_threshold,
            errors=[]
        )
    
    def _calculate_confidence(self, ocr_conf: float, format_valid: bool) -> float:
        """
        Calculate final confidence score.
        
        Args:
            ocr_conf: OCR confidence (0-1)
            format_valid: Whether format validation passed
            
        Returns:
            Combined confidence score
        """
        format_score = 1.0 if format_valid else 0.5
        
        # Weighted average: 40% OCR, 60% format
        confidence = (ocr_conf * 0.4) + (format_score * 0.6)
        
        return round(confidence, 2)
    
    def validate_all_fields(
        self, 
        fields: Dict[str, Any], 
        ocr_confidences: Optional[Dict[str, float]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Validate all extracted fields.
        
        Args:
            fields: Dictionary of field_name: value
            ocr_confidences: Optional dictionary of field_name: confidence
            
        Returns:
            Dictionary of field_name: ValidationResult
        """
        ocr_confidences = ocr_confidences or {}
        results = {}
        
        for field_name, value in fields.items():
            ocr_conf = ocr_confidences.get(field_name, 0.8)  # Default confidence
            results[field_name] = self.validate_field(field_name, value, ocr_conf)
        
        return results
    
    def cross_validate(self, fields: Dict[str, Any]) -> List[str]:
        """
        Perform cross-field validation for logical consistency.
        
        Args:
            fields: Dictionary of extracted fields
            
        Returns:
            List of inconsistency warnings
        """
        warnings = []
        
        # Age vs DOB check
        if 'age' in fields and 'date_of_birth' in fields:
            try:
                dob = fields['date_of_birth']
                if isinstance(dob, str):
                    dob = datetime.fromisoformat(dob).date()
                
                calculated_age = (date.today() - dob).days // 365
                stated_age = int(fields['age'])
                
                if abs(calculated_age - stated_age) > 1:
                    warnings.append(f"Age ({stated_age}) doesn't match DOB (calculated: {calculated_age})")
            except (ValueError, TypeError):
                pass
        
        # Address consistency: city should be in full address
        if 'city' in fields and 'permanent_address' in fields:
            if fields['city'] and fields['permanent_address']:
                if fields['city'].lower() not in fields['permanent_address'].lower():
                    warnings.append("City may not match address")
        
        return warnings
    
    def get_fields_needing_review(self, results: Dict[str, ValidationResult]) -> List[str]:
        """
        Get list of fields that need human review.
        
        Args:
            results: Validation results
            
        Returns:
            List of field names needing review
        """
        return [name for name, result in results.items() if result.needs_review]
