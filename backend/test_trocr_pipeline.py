"""
Quick test script for TrOCR with line segmentation.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

def test_line_segmentation():
    """Test line segmentation on a sample image."""
    print("=" * 60)
    print("Testing Line Segmentation")
    print("=" * 60)
    
    from modules.line_segmentor import SSDLineSegmentor
    
    # Create test image with text-like content
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some text lines
    cv2.putText(test_image, "Line 1: Hello World", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, "Line 2: Testing OCR", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, "Line 3: Form Extraction", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, "Line 4: TrOCR Test", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Test segmentor
    segmentor = SSDLineSegmentor(min_line_height=5, min_line_width=30)
    lines = segmentor.detect_lines(test_image)
    
    print(f"Detected {len(lines)} lines:")
    for i, line in enumerate(lines):
        print(f"  Line {i+1}: x={line.x}, y={line.y}, w={line.width}, h={line.height}")
    
    # Test segmentation
    segments = segmentor.segment_lines(test_image, lines)
    print(f"\nSegmented into {len(segments)} line images")
    
    return len(lines) > 0


def test_trocr_engine():
    """Test TrOCR engine loading."""
    print("\n" + "=" * 60)
    print("Testing TrOCR Engine")
    print("=" * 60)
    
    try:
        from modules.trocr_engine import TrOCREngine
        
        print("Creating TrOCR engine...")
        engine = TrOCREngine()
        
        print("Initializing model (this may download ~300MB on first run)...")
        engine._ensure_initialized()
        
        info = engine.get_model_info()
        print(f"Model: {info['model_name']}")
        print(f"Device: {info['device']}")
        print(f"Parameters: {info['num_parameters']:,}")
        
        # Create a simple test image
        test_image = np.ones((50, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Hello World", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        print("\nRecognizing test image...")
        result = engine.recognize_line(test_image)
        print(f"Recognized: '{result.text}'")
        print(f"Confidence: {result.confidence:.2%}")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure transformers and torch are installed.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_integrated_ocr():
    """Test the integrated OCR engine."""
    print("\n" + "=" * 60)
    print("Testing Integrated OCR Engine")
    print("=" * 60)
    
    from modules.ocr_engine import OCREngine
    
    # Create test image
    test_image = np.ones((300, 500, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Name: John Doe", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(test_image, "Date: 2024-01-18", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(test_image, "Phone: 9876543210", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    print("Creating OCR engine...")
    ocr = OCREngine(use_trocr=True, use_tesseract_fallback=True)
    
    print("Processing image...")
    result = ocr.process_image(test_image)
    
    print(f"\nOCR Engine Used: {result.ocr_engine_used}")
    print(f"Average Confidence: {result.average_confidence:.2%}")
    print(f"Text Boxes: {len(result.text_boxes)}")
    print("\nExtracted Text:")
    print("-" * 40)
    print(result.full_text)
    print("-" * 40)
    
    return True


if __name__ == "__main__":
    print("TrOCR with Line Segmentation - Test Suite")
    print("=" * 60)
    
    # Test 1: Line segmentation
    seg_ok = test_line_segmentation()
    
    # Test 2: TrOCR engine
    trocr_ok = test_trocr_engine()
    
    # Test 3: Integrated OCR
    integrated_ok = test_integrated_ocr()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Line Segmentation: {'PASS' if seg_ok else 'FAIL'}")
    print(f"TrOCR Engine: {'PASS' if trocr_ok else 'FAIL'}")
    print(f"Integrated OCR: {'PASS' if integrated_ok else 'FAIL'}")
    
    if seg_ok and trocr_ok and integrated_ok:
        print("\nAll tests passed!")
    else:
        print("\nSome tests failed. Check the output above.")
