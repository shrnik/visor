#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import chromadb
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import plotly.graph_objects as go
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaUMAPVisualizer:
    def __init__(self, chroma_path: str = "./chroma-data"):
        """Initialize the visualizer with ChromaDB connection."""
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = None
        self.embeddings = None
        self.metadata = None
        self.documents = None
        self.umap_embeddings = None
        
    def connect_to_collection(self, collection_name: str = "east_camera") -> bool:
        """Connect to the specified ChromaDB collection."""
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Connected to collection: {collection_name}")
            logger.info(f"Collection has {self.collection.count()} items")
            return True
        except Exception as e:
            logger.error(f"Error connecting to collection {collection_name}: {e}")
            return False
    
    def extract_data(self, limit: Optional[int] = None) -> bool:
        """Extract embeddings, metadata, and documents from the collection."""
        try:
            if not self.collection:
                logger.error("No collection connected")
                return False
            
            results = self.collection.get(
                limit=limit,
                include=["embeddings", "metadatas", "documents"]
            )
            
            # if not results.get('embeddings'):
            #     logger.error("No embeddings found in collection")
            #     return False
            
            self.embeddings = np.array(results['embeddings'])
            self.metadata = results.get('metadatas', [])
            self.documents = results.get('documents', [])
            
            logger.info(f"Extracted {len(self.embeddings)} embeddings")
            logger.info(f"Embedding dimension: {self.embeddings.shape[1]}")
            
            return True
        except Exception as e:
            logger.error(f"Error extracting data: {e}")
            return False
    
    def apply_umap(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                   n_components: int = 2, metric: str = 'cosine', 
                   random_state: int = 42) -> bool:
        """Apply UMAP dimensionality reduction to embeddings."""
        try:
            if self.embeddings is None:
                logger.error("No embeddings available for UMAP")
                return False
            
            logger.info("Applying UMAP dimensionality reduction...")
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=n_components,
                metric=metric,
                random_state=random_state,
                verbose=True
            )
            
            self.umap_embeddings = reducer.fit_transform(self.embeddings)
            logger.info(f"UMAP completed. Output shape: {self.umap_embeddings.shape}")
            
            return True
        except Exception as e:
            logger.error(f"Error applying UMAP: {e}")
            return False
    
    def create_interactive_plot(self, save_path: str = "./visualizations/umap_east_camera.html") -> bool:
        """Create an interactive plotly visualization of UMAP embeddings."""
        try:
            if self.umap_embeddings is None:
                logger.error("No UMAP embeddings available for plotting")
                return False
            
            # Prepare data for plotting
            df = pd.DataFrame({
                'UMAP1': self.umap_embeddings[:, 0],
                'UMAP2': self.umap_embeddings[:, 1]
            })
            
            # Add metadata if available
            if self.metadata:
                for i, meta in enumerate(self.metadata):
                    if meta:
                        for key, value in meta.items():
                            if key not in df.columns:
                                df[key] = None
                            df.loc[i, key] = value
            
            # Add document preview if available
            if self.documents:
                df['document_preview'] = [
                    str(doc)[:100] + "..." if len(str(doc)) > 100 else str(doc)
                    for doc in self.documents
                ]
            
            # Create hover text
            hover_data = []
            for i in range(len(df)):
                hover_text = f"Point {i}<br>"
                hover_text += f"UMAP1: {df.iloc[i]['UMAP1']:.3f}<br>"
                hover_text += f"UMAP2: {df.iloc[i]['UMAP2']:.3f}<br>"
                
                if 'document_preview' in df.columns:
                    hover_text += f"Document: {df.iloc[i]['document_preview']}<br>"
                
                # Add any other metadata
                for col in df.columns:
                    if col not in ['UMAP1', 'UMAP2', 'document_preview'] and pd.notna(df.iloc[i][col]):
                        hover_text += f"{col}: {df.iloc[i][col]}<br>"
                
                hover_data.append(hover_text)
            
            df['hover_text'] = hover_data
            
            # Create the plot
            fig = go.Figure()
            
            # Color by cluster if we have many points
            if len(df) > 50:
                # Simple clustering based on spatial proximity for coloring
                from sklearn.cluster import KMeans
                n_clusters = min(10, len(df) // 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                df['cluster'] = kmeans.fit_predict(self.umap_embeddings)
                
                colors = px.colors.qualitative.Set3[:n_clusters]
                
                for cluster_id in range(n_clusters):
                    cluster_data = df[df['cluster'] == cluster_id]
                    fig.add_trace(go.Scatter(
                        x=cluster_data['UMAP1'],
                        y=cluster_data['UMAP2'],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=colors[cluster_id],
                            opacity=0.7,
                            line=dict(width=1, color='white')
                        ),
                        text=cluster_data['hover_text'],
                        hovertemplate='%{text}<extra></extra>',
                        name=f'Cluster {cluster_id}'
                    ))
            else:
                # Simple scatter plot for smaller datasets
                fig.add_trace(go.Scatter(
                    x=df['UMAP1'],
                    y=df['UMAP2'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='lightblue',
                        opacity=0.7,
                        line=dict(width=2, color='darkblue')
                    ),
                    text=df['hover_text'],
                    hovertemplate='%{text}<extra></extra>',
                    name='Embeddings'
                ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': 'Interactive UMAP Visualization - East Camera Collection',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                width=1000,
                height=700,
                hovermode='closest',
                showlegend=True,
                template='plotly_white'
            )
            
            # Add some styling
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
            # Save the plot
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to: {save_path}")
            
            # Also show the plot in browser
            fig.show()
            
            return True
        except Exception as e:
            logger.error(f"Error creating interactive plot: {e}")
            return False
    
    def run_complete_visualization(self, collection_name: str = "east_camera", 
                                 limit: Optional[int] = None,
                                 save_path: str = "./visualizations/umap_east_camera.html"):
        """Run the complete visualization pipeline."""
        logger.info("Starting UMAP visualization pipeline...")
        
        if not self.connect_to_collection(collection_name):
            return False
        
        if not self.extract_data(limit):
            return False
        
        if not self.apply_umap():
            return False
        
        if not self.create_interactive_plot(save_path):
            return False
        
        logger.info("Visualization pipeline completed successfully!")
        return True

def main():
    """Main function to run the UMAP visualization."""
    visualizer = ChromaUMAPVisualizer()
    
    # Run the complete visualization
    success = visualizer.run_complete_visualization(
        collection_name="east_camera",
        limit=None,  # Use all data, set to a number to limit
        save_path="./visualizations/umap_east_camera.html"
    )
    
    if success:
        print("\nUMAP visualization completed successfully!")
        print("Interactive plot saved to: ./visualizations/umap_east_camera.html")
        print("The plot should have opened in your browser automatically")
    else:
        print("\nFailed to create UMAP visualization")
        print("Check the logs above for error details")

if __name__ == "__main__":
    main()