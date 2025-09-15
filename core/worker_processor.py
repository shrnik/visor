import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
from core.queue_config import insertion_queue

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
processor = None

def load_siglip_model():
    global model, processor
    if model is None or processor is None:
        print("Loading SigLip2 model...")
        model_name = "google/siglip-base-patch16-384"
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        print(f"Model loaded on device: {device}")

def process_image_embedding(image_data):
    try:
        load_siglip_model()
        
        local_path = image_data['local_path']
        
        if not os.path.exists(local_path):
            raise Exception(f"Image file not found: {local_path}")
        
        with Image.open(local_path) as img:
            img = img.convert('RGB')
            
            inputs = processor(images=img, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
                
                embedding = outputs.cpu().numpy().flatten()
        
        embedding_data = {
            'filename': image_data['filename'],
            'local_path': local_path,
            'url': image_data['url'],
            'embedding': embedding.tolist(),
            'embedding_model': 'google/siglip-base-patch16-384',
            'embedding_dim': len(embedding)
        }
        
        insertion_job = insertion_queue.enqueue(
            'core.worker_insertion.insert_embedding',
            embedding_data,
            timeout=300
        )
        
        print(f"Processed embedding for {image_data['filename']}, enqueued for insertion: {insertion_job.id}")
        return embedding_data
        
    except Exception as e:
        print(f"Error processing embedding for {image_data.get('filename', 'unknown')}: {e}")
        return {
            'success': False,
            'error': str(e),
            'filename': image_data.get('filename', 'unknown')
        }

if __name__ == "__main__":
    from rq import Worker
    from core.queue_config import processing_queue, redis_conn
    
    print("Starting processing worker...")
    worker = Worker([processing_queue], connection=redis_conn)
    worker.work()