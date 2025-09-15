# Weather Camera Image Processing Pipeline

A distributed processing pipeline that scrapes weather camera images, computes SigLip2 embeddings, and stores them in ChromaDB using Redis Queue (RQ).

## Prerequisites

1. Install Redis server:
   ```bash
   # macOS
   brew install redis
   
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # Start Redis
   redis-server
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Overview

The pipeline consists of three workers:

1. **Scraper/Downloader Worker**: Scrapes image links from the weather camera website and downloads images
2. **Processing Worker**: Computes SigLip2 embeddings for downloaded images
3. **Insertion Worker**: Stores embeddings and metadata in ChromaDB

## Usage

### 1. Start Redis (in a separate terminal)
```bash
redis-server
```

### 2. Start Workers (each in separate terminals)

Start the download worker:
```bash
python -m core.main download-worker
```

Start the processing worker:
```bash
python -m core.main processing-worker
```

Start the insertion worker:
```bash
python -m core.main insertion-worker
```

Alternatively, start all workers in one process:
```bash
python -m core.main all-workers
```

### 3. Initiate Scraping
```bash
python -m core.main scrape
```

### 4. Monitor Progress
```bash
python -m core.main status
```

### 5. Clear Queues (if needed)
```bash
python -m core.main clear
```

## Queue Flow

```
Scraper → download_queue → Processing → processing_queue → Insertion → insertion_queue → ChromaDB
```

## File Structure

```
core/
├── README.md              # This file
├── Plan.md               # Original requirements
├── main.py               # Main orchestration script
├── queue_config.py       # RQ queue configuration
├── worker_scraper.py     # Image scraping and downloading
├── worker_processor.py   # SigLip2 embedding computation
└── worker_insertion.py   # ChromaDB insertion
```

## Environment Variables

- `REDIS_HOST`: Redis host (default: localhost)
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_DB`: Redis database (default: 0)
- `REDIS_PASSWORD`: Redis password (default: None)

## Notes

- Images are downloaded to `downloaded_images/` directory
- ChromaDB data is stored in `chroma-data/` directory
- The pipeline uses SigLip2 model: `google/siglip-base-patch16-384`
- Each worker can be scaled independently by running multiple instances