from database.mongo_client import MongoDBClient
import json
from bson import ObjectId

def default(o):
    if isinstance(o, ObjectId):
        return str(o)
    return str(o)

print("Connecting to MongoDB...")
client = MongoDBClient()
results = client.get_all_extractions(limit=1)

if results:
    doc = results[0]
    print(f"\nLatest Document ID: {doc.get('_id')}")
    print("Top-level keys:", list(doc.keys()))
    
    if 'fields' in doc:
        print(f"\n'fields' count: {len(doc.get('fields', {}))}")
        print("'fields' content:", json.dumps(doc.get('fields'), default=default, indent=2)[:500])
        
    if 'raw_fields' in doc:
        print(f"\n'raw_fields' count: {len(doc.get('raw_fields', {}))}")
    
    if 'extracted_fields' in doc:
        print(f"\nFound 'extracted_fields' key!")
    else:
        print(f"\nKey 'extracted_fields' NOT found in document.")
else:
    print("No documents found.")
