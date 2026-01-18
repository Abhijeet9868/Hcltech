"""
MongoDB Client Module for Form Extraction.
Handles database operations for storing and retrieving extractions.
"""

from pymongo import MongoClient
from pymongo.collection import Collection
from bson.objectid import ObjectId
from datetime import datetime
from typing import Dict, List, Optional, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class MongoDBClient:
    """MongoDB client for form extraction data storage."""
    
    def __init__(self, uri: Optional[str] = None, db_name: Optional[str] = None):
        """
        Initialize MongoDB client.
        
        Args:
            uri: MongoDB connection URI
            db_name: Database name
        """
        self.uri = uri or Config.MONGODB_URI
        self.db_name = db_name or Config.MONGODB_DB_NAME
        
        self.client = MongoClient(self.uri)
        self.db = self.client[self.db_name]
        
        # Collections
        self.extractions: Collection = self.db['extractions']
        self.forms: Collection = self.db['forms']
    
    def save_extraction(self, form_data: Dict[str, Any]) -> str:
        """
        Save extraction result to database.
        
        Args:
            form_data: Extracted form data including fields, confidence, etc.
            
        Returns:
            Document ID as string
        """
        document = {
            'filename': form_data.get('filename', 'unknown'),
            'upload_date': datetime.utcnow(),
            'form_type': form_data.get('form_type', 'banking_application'),
            'extracted_fields': form_data.get('fields', {}),
            'confidence_scores': form_data.get('confidence_scores', {}),
            'validation_results': form_data.get('validation_results', {}),
            'needs_review': form_data.get('needs_review', []),
            'processing_time_ms': form_data.get('processing_time_ms', 0),
            'ocr_text': form_data.get('ocr_text', ''),
            'status': form_data.get('status', 'completed'),
            'has_signature': form_data.get('has_signature', False),
            'checkboxes': form_data.get('checkboxes', [])
        }
        
        result = self.extractions.insert_one(document)
        return str(result.inserted_id)
    
    def get_extraction(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve extraction by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Extraction document or None
        """
        try:
            result = self.extractions.find_one({'_id': ObjectId(doc_id)})
            if result:
                result['_id'] = str(result['_id'])  # Convert ObjectId to string
            return result
        except Exception:
            return None
    
    def get_all_extractions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve all extractions, most recent first.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of extraction documents
        """
        results = list(
            self.extractions.find()
            .sort('upload_date', -1)
            .limit(limit)
        )
        
        # Convert ObjectIds to strings
        for doc in results:
            doc['_id'] = str(doc['_id'])
        
        return results
    
    def update_extraction(self, doc_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update an existing extraction.
        
        Args:
            doc_id: Document ID
            update_data: Fields to update
            
        Returns:
            True if update was successful
        """
        try:
            result = self.extractions.update_one(
                {'_id': ObjectId(doc_id)},
                {'$set': update_data}
            )
            return result.modified_count > 0
        except Exception:
            return False
    
    def delete_extraction(self, doc_id: str) -> bool:
        """
        Delete an extraction.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if deletion was successful
        """
        try:
            result = self.extractions.delete_one({'_id': ObjectId(doc_id)})
            return result.deleted_count > 0
        except Exception:
            return False
    
    def search_extractions(
        self, 
        query: Dict[str, Any], 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search extractions with custom query.
        
        Args:
            query: MongoDB query dictionary
            limit: Maximum results
            
        Returns:
            List of matching documents
        """
        results = list(
            self.extractions.find(query)
            .sort('upload_date', -1)
            .limit(limit)
        )
        
        for doc in results:
            doc['_id'] = str(doc['_id'])
        
        return results
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about extractions.
        
        Returns:
            Statistics dictionary
        """
        total = self.extractions.count_documents({})
        
        # Count by form type
        form_types = self.extractions.aggregate([
            {'$group': {'_id': '$form_type', 'count': {'$sum': 1}}}
        ])
        
        # Average confidence
        avg_confidence = self.extractions.aggregate([
            {'$group': {'_id': None, 'avg': {'$avg': '$confidence_scores.overall'}}}
        ])
        
        return {
            'total_extractions': total,
            'form_types': {doc['_id']: doc['count'] for doc in form_types},
            'needs_review_count': self.extractions.count_documents({'needs_review': {'$ne': []}})
        }
    
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()
