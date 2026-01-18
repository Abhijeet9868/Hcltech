import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration settings."""
    
    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "form_extraction")
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    # Tesseract Configuration
    TESSERACT_PATH = os.getenv("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    
    # Upload Configuration
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    
    # OCR Configuration
    OCR_LANGUAGE = "eng"
    OCR_CONFIG = "--psm 6"  # Assume uniform block of text
    
    # TrOCR Configuration
    # Note: TrOCR is slow on CPU (~10-20s per line). Enable when GPU is available.
    # Set to "auto" to enable only when CUDA is available
    USE_TROCR = os.getenv("USE_TROCR", "auto").lower()  # "true", "false", or "auto"
    TROCR_MODEL = os.getenv("TROCR_MODEL", "microsoft/trocr-base-handwritten")
    TROCR_DEVICE = os.getenv("TROCR_DEVICE", "auto")  # auto, cuda, cpu
    TROCR_BATCH_SIZE = int(os.getenv("TROCR_BATCH_SIZE", "8"))
    
    # Line Segmentation Configuration
    LINE_DETECTION_CONFIDENCE = float(os.getenv("LINE_DETECTION_CONFIDENCE", "0.5"))
    USE_SSD_LINE_DETECTION = os.getenv("USE_SSD_LINE_DETECTION", "false").lower() == "true"
    
    # Fallback Configuration
    USE_TESSERACT_FALLBACK = os.getenv("USE_TESSERACT_FALLBACK", "true").lower() == "true"
    
    # Confidence Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    LOW_CONFIDENCE_THRESHOLD = 0.6

