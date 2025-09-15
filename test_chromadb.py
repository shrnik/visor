#!/usr/bin/env python3

import chromadb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chromadb():
    try:
        client = chromadb.PersistentClient(path="./chroma-data")
        logger.info("Connected to ChromaDB")
        
        # List all collections
        collections = client.list_collections()
        logger.info(f"Found collections: {[c.name for c in collections]}")
        
        # Try to get the collection
        try:
            collection = client.get_collection("east_camera")
            logger.info(f"Found collection: {collection.name}")
            
            # Get count
            count = collection.count()
            logger.info(f"Collection has {count} items")
            
            if count > 0:
                # Get a small sample
                results = collection.get(limit=3, include=["embeddings", "metadatas", "documents"])
                logger.info(f"Sample results keys: {list(results.keys())}")
                
                if results.get('embeddings') and len(results['embeddings']) > 0:
                    logger.info(f"Sample embedding shape: {len(results['embeddings'][0])}")
                else:
                    logger.info("Sample embedding shape: None")
                    
                if results.get('metadatas') and len(results['metadatas']) > 0:
                    logger.info(f"Sample metadata: {results['metadatas'][0]}")
                else:
                    logger.info("Sample metadata: None")
                
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            
    except Exception as e:
        logger.error(f"Error connecting to ChromaDB: {e}")

if __name__ == "__main__":
    test_chromadb()