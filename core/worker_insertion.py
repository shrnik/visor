import chromadb
import os
from datetime import datetime
import hashlib
import re


chroma_client = None
collection = None

def initialize_chromadb():
    global chroma_client, collection
    if chroma_client is None:
        chroma_client = chromadb.PersistentClient(path="./chroma-data")
        collection = chroma_client.get_or_create_collection(
            name="image_embeddings",
            metadata={"description": "SigLip2 embeddings of weather camera images"}
        )
        print("ChromaDB initialized")

def insert_embedding(embedding_data):
    try:
        initialize_chromadb()
        url = embedding_data['url']
        timestamp = embedding_data.get('timestamp', 'unknown')
        embedding = embedding_data['embedding']
        doc_id = f"weather_img_{timestamp}"
        metadata = {
            'url': url,
            'embedding_model': embedding_data.get('embedding_model', 'unknown'),
            'inserted_at': datetime.now().isoformat(),
            'timestamp': timestamp,
        }
        
        collection.upsert(
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[f"Weather camera image: {timestamp}"],
            ids=[doc_id]
        )

        print(f"Inserted embedding for {timestamp} with ID {doc_id}")

        return {
            'success': True,
            'document_id': doc_id   ,
            'filename': url,
            'collection_count': collection.count()
        }
        
    except Exception as e:
        print(f"Error inserting embedding for {embedding_data.get('filename', 'unknown')}: {e}")
        return {
            'success': False,
            'error': str(e),
            'filename': embedding_data.get('filename', 'unknown')
        }

def get_collection_stats():
    try:
        initialize_chromadb()
        count = collection.count()
        return {
            'collection_name': 'image_embeddings',
            'document_count': count
        }
    except Exception as e:
        return {
            'error': str(e)
        }

if __name__ == "__main__":
    from rq import Worker
    from core.queue_config import insertion_queue, redis_conn
    
    print("Starting insertion worker...")
    worker = Worker([insertion_queue], connection=redis_conn)
    worker.work()