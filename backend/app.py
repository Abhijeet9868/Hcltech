"""
Form Extraction API - Flask Backend
Handles form upload, OCR processing, and data extraction.
"""

import os
import time
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from io import BytesIO
import csv

from config import Config
from modules.preprocessing import ImagePreprocessor
from modules.ocr_engine import OCREngine
from modules.layout_detector import LayoutDetector
from modules.llm_extractor import LLMExtractor
from modules.validator import FieldValidator
from database.mongo_client import MongoDBClient


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)

# Initialize components
preprocessor = ImagePreprocessor()
ocr_engine = OCREngine()
layout_detector = LayoutDetector()
validator = FieldValidator()

# MongoDB client (initialized lazily)
db_client = None

def get_db():
    """Get MongoDB client instance."""
    global db_client
    if db_client is None:
        try:
            db_client = MongoDBClient()
        except Exception as e:
            print(f"Warning: MongoDB connection failed: {e}")
            return None
    return db_client


def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON/MongoDB serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    })


@app.route('/api/upload', methods=['POST'])
def upload_form():
    """
    Upload a form image for processing.
    Returns a job ID for tracking.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, pdf'}), 400
    
    # Save file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saved_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, saved_filename)
    file.save(filepath)
    
    return jsonify({
        'message': 'File uploaded successfully',
        'filename': saved_filename,
        'filepath': filepath
    })


@app.route('/api/extract', methods=['POST'])
def extract_form_data():
    """
    Extract data from an uploaded form.
    Accepts either a file upload or a filepath.
    """
    start_time = time.time()
    
    # Get image from request
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read image from file
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        filename = secure_filename(file.filename)
    elif request.is_json and 'filepath' in request.json:
        filepath = request.json['filepath']
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        image = cv2.imread(filepath)
        filename = os.path.basename(filepath)
    else:
        return jsonify({'error': 'No file or filepath provided'}), 400
    
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400
    
    try:
        # Step 1: Preprocess image
        preprocessed = preprocessor.preprocess(image)
        
        # Step 2: OCR extraction
        ocr_result = ocr_engine.process_image(preprocessed)
        
        # Step 3: Layout detection
        key_value_pairs = layout_detector.detect_key_value_pairs(ocr_result.text_boxes)
        
        # Convert to dict for LLM context
        pairs_dict = [{'key': p.key, 'value': p.value} for p in key_value_pairs]
        
        # Step 4: LLM extraction (with fallback)
        try:
            llm_extractor = LLMExtractor()
            llm_result = llm_extractor.extract_with_ocr_context(
                image, ocr_result.full_text, pairs_dict
            )
            extracted_fields = llm_result.fields
            form_type = llm_result.form_type
            llm_confidence = llm_result.confidence
        except Exception as e:
            # Fallback to rule-based extraction if LLM fails
            print(f"LLM extraction failed: {e}")
            extracted_fields = {p.key: p.value for p in key_value_pairs}
            form_type = 'banking_application'
            llm_confidence = 0.6
        
        # Step 5: Validate extracted fields
        validation_results = validator.validate_all_fields(extracted_fields)
        
        # Build confidence scores
        confidence_scores = {
            field: result.confidence 
            for field, result in validation_results.items()
        }
        confidence_scores['overall'] = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
        
        # Get fields needing review
        needs_review = validator.get_fields_needing_review(validation_results)
        
        # Cross-validation warnings
        cross_validation_warnings = validator.cross_validate(extracted_fields)
        
        # Normalize fields using validation results
        normalized_fields = {
            field: result.normalized_value 
            for field, result in validation_results.items()
        }
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Prepare result
        result = {
            'filename': filename,
            'form_type': form_type,
            'fields': normalized_fields,
            'raw_fields': extracted_fields,
            'confidence_scores': confidence_scores,
            'validation_results': {
                field: {
                    'is_valid': r.is_valid,
                    'confidence': r.confidence,
                    'errors': r.errors
                } for field, r in validation_results.items()
            },
            'needs_review': needs_review,
            'cross_validation_warnings': cross_validation_warnings,
            'ocr_text': ocr_result.full_text,
            'ocr_confidence': ocr_result.average_confidence,
            'has_signature': ocr_result.has_signature,
            'checkboxes': [
                {'x': cb.x, 'y': cb.y, 'checked': cb.is_checked}
                for cb in ocr_result.checkboxes
            ],
            'processing_time_ms': processing_time,
            'status': 'completed'
        }
        
        # Convert numpy types to Python native types
        result = convert_numpy_types(result)
        
        # Save to database
        db = get_db()
        if db:
            doc_id = db.save_extraction(result)
            result['_id'] = doc_id
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500


@app.route('/api/results/<doc_id>', methods=['GET'])
def get_extraction_result(doc_id):
    """Get extraction result by ID."""
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database not available'}), 503
    
    result = db.get_extraction(doc_id)
    if result is None:
        return jsonify({'error': 'Result not found'}), 404
    
    return jsonify(result)


@app.route('/api/history', methods=['GET'])
def get_extraction_history():
    """Get all extraction history."""
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database not available'}), 503
    
    limit = request.args.get('limit', 100, type=int)
    results = db.get_all_extractions(limit=limit)
    
    return jsonify({
        'total': len(results),
        'results': results
    })


@app.route('/api/results/<doc_id>', methods=['PUT'])
def update_extraction(doc_id):
    """Update extraction (for manual corrections)."""
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database not available'}), 503
    
    update_data = request.json
    success = db.update_extraction(doc_id, update_data)
    
    if success:
        return jsonify({'message': 'Updated successfully'})
    return jsonify({'error': 'Update failed'}), 400


@app.route('/api/results/<doc_id>', methods=['DELETE'])
def delete_extraction(doc_id):
    """Delete an extraction."""
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database not available'}), 503
    
    success = db.delete_extraction(doc_id)
    
    if success:
        return jsonify({'message': 'Deleted successfully'})
    return jsonify({'error': 'Delete failed'}), 400


@app.route('/api/export/<doc_id>/<format>', methods=['GET'])
def export_extraction(doc_id, format):
    """Export extraction as JSON or CSV."""
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database not available'}), 503
    
    result = db.get_extraction(doc_id)
    if result is None:
        return jsonify({'error': 'Result not found'}), 404
    
    if format == 'json':
        return jsonify(result)
    
    elif format == 'csv':
        # Convert to CSV
        output = BytesIO()
        fields = result.get('fields', {})
        
        writer = csv.writer(output)
        writer.writerow(['Field', 'Value', 'Confidence', 'Valid'])
        
        for field, value in fields.items():
            conf = result.get('confidence_scores', {}).get(field, 'N/A')
            valid = result.get('validation_results', {}).get(field, {}).get('is_valid', 'N/A')
            writer.writerow([field, value, conf, valid])
        
        output.seek(0)
        return send_file(
            BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'extraction_{doc_id}.csv'
        )
    
    return jsonify({'error': 'Invalid format. Use json or csv'}), 400


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get extraction statistics."""
    db = get_db()
    if db is None:
        return jsonify({'error': 'Database not available'}), 503
    
    stats = db.get_extraction_stats()
    return jsonify(stats)


if __name__ == '__main__':
    print("Starting Form Extraction API...")
    print(f"Upload folder: {Config.UPLOAD_FOLDER}")
    print(f"MongoDB URI: {Config.MONGODB_URI}")
    app.run(debug=True, host='0.0.0.0', port=5000)
