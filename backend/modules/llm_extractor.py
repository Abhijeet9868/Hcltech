"""
LLM Extractor Module for Form Extraction.
Uses OpenAI GPT-4 Vision for intelligent field extraction.
"""

import base64
import json
import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


@dataclass
class ExtractionResult:
    """Result from LLM-based extraction."""
    fields: Dict[str, Any]
    form_type: str
    confidence: float
    raw_response: str


class LLMExtractor:
    """Uses OpenAI GPT-4 Vision for intelligent form field extraction."""
    
    # Banking form extraction prompt
    BANKING_FORM_PROMPT = """Analyze this banking application form image and extract all relevant information.

Extract the following fields if present (use null if not found):
- full_name: Applicant's full name
- date_of_birth: Date of birth (format as YYYY-MM-DD if possible)
- gender: Gender
- nationality: Nationality
- marital_status: Marital status
- father_name: Father's name
- mother_name: Mother's name
- permanent_address: Full permanent address
- current_address: Full current address
- city: City
- state: State
- pincode: PIN/ZIP code
- phone: Phone/Mobile number
- email: Email address
- pan_number: PAN card number
- aadhaar_number: Aadhaar number (12 digits)
- occupation: Occupation/Profession
- employer_name: Employer/Company name
- annual_income: Annual income
- account_type: Type of account (Savings/Current/etc.)
- nominee_name: Nominee's name
- nominee_relationship: Relationship with nominee

Also identify:
- form_type: Type of form (e.g., "savings_account_opening", "loan_application", "kyc_update")
- has_signature: Whether a signature is present (true/false)
- checkboxes: List any checked checkbox options

Return the result as a valid JSON object with this exact structure:
{
  "fields": {
    "full_name": "...",
    "date_of_birth": "...",
    ... (all fields listed above)
  },
  "form_type": "...",
  "has_signature": true/false,
  "checkboxes": ["option1", "option2"],
  "extraction_confidence": 0.0-1.0
}

Be thorough and extract all visible information. If a field is partially visible or unclear, still attempt to extract it but note the uncertainty in the confidence score."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize LLM Extractor.
        
        Args:
            api_key: OpenAI API key (uses config if not provided)
            model: Model to use for extraction
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY in .env")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def _encode_image(self, image: np.ndarray) -> str:
        """
        Encode image to base64 for API request.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Base64 encoded string
        """
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', image)
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image
    
    def _encode_image_file(self, file_path: str) -> str:
        """
        Encode image file to base64.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def extract_fields(self, image: np.ndarray, custom_prompt: Optional[str] = None) -> ExtractionResult:
        """
        Extract form fields using GPT-4 Vision.
        
        Args:
            image: Input form image
            custom_prompt: Optional custom extraction prompt
            
        Returns:
            ExtractionResult with extracted fields
        """
        # Encode image
        base64_image = self._encode_image(image)
        
        # Use custom prompt or default banking prompt
        prompt = custom_prompt or self.BANKING_FORM_PROMPT
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1  # Low temperature for consistent extraction
            )
            
            # Parse response
            raw_response = response.choices[0].message.content
            
            # Try to extract JSON from response
            extracted_data = self._parse_json_response(raw_response)
            
            return ExtractionResult(
                fields=extracted_data.get('fields', {}),
                form_type=extracted_data.get('form_type', 'unknown'),
                confidence=extracted_data.get('extraction_confidence', 0.8),
                raw_response=raw_response
            )
            
        except Exception as e:
            # Return empty result on error
            return ExtractionResult(
                fields={},
                form_type='unknown',
                confidence=0.0,
                raw_response=f"Error: {str(e)}"
            )
    
    def _parse_json_response(self, response: str) -> Dict:
        """
        Parse JSON from LLM response, handling markdown code blocks.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed dictionary
        """
        # Remove markdown code blocks if present
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            response = response[start:end].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON object in response
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except json.JSONDecodeError:
                    pass
            return {}
    
    def classify_form_type(self, text: str) -> str:
        """
        Classify form type based on extracted text.
        
        Args:
            text: OCR text from form
            
        Returns:
            Form type classification
        """
        text_lower = text.lower()
        
        # Banking form classifications
        if any(kw in text_lower for kw in ['savings account', 'account opening']):
            return 'savings_account_opening'
        elif any(kw in text_lower for kw in ['loan application', 'loan form']):
            return 'loan_application'
        elif any(kw in text_lower for kw in ['kyc', 'know your customer']):
            return 'kyc_update'
        elif any(kw in text_lower for kw in ['fixed deposit', 'fd application']):
            return 'fixed_deposit'
        elif any(kw in text_lower for kw in ['credit card', 'card application']):
            return 'credit_card_application'
        
        return 'banking_application'
    
    def extract_with_ocr_context(
        self, 
        image: np.ndarray, 
        ocr_text: str, 
        key_value_pairs: List[Dict]
    ) -> ExtractionResult:
        """
        Extract fields using both image and OCR context.
        
        Args:
            image: Form image
            ocr_text: Pre-extracted OCR text
            key_value_pairs: Pre-detected key-value pairs
            
        Returns:
            Enhanced extraction result
        """
        # Build enhanced prompt with OCR context
        context_prompt = f"""Analyze this banking form. I have already extracted some text and key-value pairs:

OCR Text:
{ocr_text[:2000]}  # Limit context size

Detected Key-Value Pairs:
{json.dumps(key_value_pairs[:20], indent=2)}

Using both the image and this context, extract all form fields accurately.
Correct any OCR errors you notice by comparing with the image.

{self.BANKING_FORM_PROMPT}"""
        
        return self.extract_fields(image, custom_prompt=context_prompt)
    
    def generate_summary(self, extracted_fields: Dict[str, Any]) -> str:
        """
        Generate a summary of the extracted form data using LLM.
        
        Args:
            extracted_fields: Dictionary of extracted fields
            
        Returns:
            Generated summary text
        """
        # Create a text representation of the fields
        if not extracted_fields:
            return "No data extracted from this form to summarize."
            
        fields_text = json.dumps(extracted_fields, indent=2)
        print(f"LLM Summary Input Fields: {fields_text}")
        
        prompt = f"""You are an expert banking analyst. Generate a professional summary based *strictly* on the following extracted form data.
        
        Extracted Data:
        {fields_text}
        
        Instructions:
        1. Identify the applicant's name and the specific type of form/application.
        2. Summarize key information found in the data (e.g., contact details, employment, financial figures).
        3. Do NOT hallucinate information not present in the data.
        4. If important fields (like Name) are missing, explicitly mention that they were not extracted.
        5. Format as a concise paragraph suitable for a banker to read quickly.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes form data."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"
