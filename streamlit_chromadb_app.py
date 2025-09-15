#!/usr/bin/env python3

import streamlit as st
import chromadb
import numpy as np
from transformers import AutoProcessor, AutoModel
import torch
import pandas as pd
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_embedding_model():
    """Load the SigLIP model for generating text embeddings"""
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-384")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-384")
    print("Loaded SigLIP model and processor")
    return processor, model

@st.cache_resource
def connect_to_chromadb():
    """Connect to ChromaDB and return the client"""
    try:
        client = chromadb.PersistentClient(path="./chroma-data")
        return client
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}")
        return None

def get_collections(client):
    """Get list of available collections"""
    try:
        collections = client.list_collections()
        return [c.name for c in collections]
    except Exception as e:
        st.error(f"Error getting collections: {e}")
        return []

def query_collection(collection, query_embedding, n_results=10):
    """Query the collection with an embedding"""
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["embeddings", "metadatas", "documents", "distances"]
        )
        return results
    except Exception as e:
        st.error(f"Error querying collection: {e}")
        return None

def format_results(results: Dict[str, Any]) -> pd.DataFrame:
    """Format query results into a pandas DataFrame"""
    if not results or not results.get('ids') or len(results['ids']) == 0:
        return pd.DataFrame()
    
    data = []
    ids = results['ids'][0] if results['ids'] else []
    documents = results['documents'][0] if results.get('documents') else [None] * len(ids)
    metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(ids)
    distances = results['distances'][0] if results.get('distances') else [None] * len(ids)
    
    for i, id_ in enumerate(ids):
        metadata = metadatas[i] if metadatas[i] else {}
        image_url = metadata.get('url', '') if isinstance(metadata, dict) else ''
        
        row = {
            'ID': id_,
            'Distance': distances[i] if distances[i] is not None else 'N/A',
            'Document': documents[i] if documents[i] else 'N/A',
            'Metadata': str(metadatas[i]) if metadatas[i] else '{}',
            'ImageURL': image_url
        }
        data.append(row)
    
    return pd.DataFrame(data)

def main():
    st.set_page_config(page_title="Visor", layout="wide")

    st.title("🔍 Visor")
    st.markdown("Explore your visual data with ease")
    
    # Initialize components
    client = connect_to_chromadb()
    if not client:
        st.error("Failed to connect to ChromaDB. Please check your setup.")
        return
    
    processor, model = load_embedding_model()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Collection selection
        collections = get_collections(client)
        if not collections:
            st.error("No collections found in ChromaDB")
            return
            
        selected_collection = st.selectbox(
            "Select Collection:",
            collections,
            index=0
        )
        
        # Number of results
        n_results = st.slider(
            "Number of Results:",
            min_value=1,
            max_value=50,
            value=10,
            help="Maximum number of results to return"
        )
        
        # Collection info
        if selected_collection:
            try:
                collection = client.get_collection(selected_collection)
                count = collection.count()
                st.info(f"Collection '{selected_collection}' has {count:,} items")
            except Exception as e:
                st.error(f"Error accessing collection: {e}")
                return
    
    # Main query interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Query Input")
        
        # Text input for query
        query_text = st.text_area(
            "Enter your query text:",
            height=100,
            placeholder="Type your search query here...",
            help="Enter any text to search for similar items in the collection"
        )
        
        # Query button
        if st.button("🔍 Search", type="primary", disabled=not query_text.strip()):
            if query_text.strip():
                with st.spinner("Generating embeddings and searching..."):
                    try:
                        # Generate embedding for query text using SigLIP
                        inputs = processor(text=[query_text.strip()], return_tensors="pt", padding=True)
                        with torch.no_grad():
                            text_features = model.get_text_features(**inputs)
                            # Normalize the embeddings
                            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                            query_embedding = text_features.squeeze().cpu().numpy().tolist()
                        
                        # Query the collection
                        collection = client.get_collection(selected_collection)
                        results = query_collection(collection, query_embedding, n_results)
                        
                        if results:
                            st.session_state['query_results'] = results
                            st.session_state['query_text'] = query_text.strip()
                        
                    except Exception as e:
                        st.error(f"Error during search: {e}")
    
    with col2:
        st.header("📊 Results")
        
        # Display results
        if 'query_results' in st.session_state:
            results = st.session_state['query_results']
            query_text = st.session_state.get('query_text', 'Unknown query')
            
            st.success(f"Found {len(results['ids'][0]) if results['ids'] else 0} results for: *{query_text}*")
            
            # Format and display results
            df = format_results(results)
            
            if not df.empty:
                # Display as expandable cards
                for idx, row in df.iterrows():
                    with st.expander(f"Result #{idx + 1} - Distance: {row['Distance']}", expanded=(idx < 3)):
                        st.image(row['ImageURL'], width=400)
                        # col_a, col_b = st.columns([1, 1])
                        
                        # with col_a:
                        #     st.write("**ID:**", row['ID'])
                        #     st.write("**Distance:**", row['Distance'])
                        
                        # with col_b:
                        #     if row['Document'] != 'N/A':
                        #         st.write("**Document:**")
                        #         st.code(row['Document'], language=None)
                        
                        # if row['Metadata'] != '{}':
                        #     st.write("**Metadata:**")
                        #     st.json(eval(row['Metadata']) if row['Metadata'].startswith('{') else row['Metadata'])
                
                # Download results as CSV
                # csv = df.to_csv(index=False)
                # st.download_button(
                #     label="📥 Download Results as CSV",
                #     data=csv,
                #     file_name=f"chromadb_query_results_{selected_collection}.csv",
                #     mime="text/csv"
                # )
            else:
                st.info("No results found for your query.")
        else:
            st.info("Enter a query above and click Search to see results here.")
    
    # Additional information
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Select a Collection**: Choose from available ChromaDB collections in the sidebar
        2. **Enter Query Text**: Type any text in the query box (e.g., descriptions, keywords)
        3. **Adjust Results**: Use the slider to control how many results to return
        4. **Search**: Click the search button to find similar items
        5. **Explore Results**: View results with similarity distances and metadata
        6. **Download**: Export results as CSV for further analysis
        
        The app uses semantic similarity to find items in your ChromaDB collection that are most similar to your query text.
        """)

if __name__ == "__main__":
    main()