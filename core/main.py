import argparse
from datetime import datetime
import sys
import os
import pandas as pd
from rq import SpawnWorker, Worker
from core.queue_config import download_queue, processing_queue, insertion_queue, redis_conn
from core.worker_scraper import scrape_and_enqueue_images
from core.worker_insertion import get_collection_stats

# Fix for macOS forking issue
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

def start_scraper():
    print("Starting image scraping and download process...")
    startDate = "2025/01/01"
    endDate = "2025/01/31"  # Use 2025/09/01 date for scraping
    for date in pd.date_range(startDate, endDate):
        date_str = date.strftime("%Y/%m/%d")
        print(f"Scraping images for date: {date_str}")
        count = scrape_and_enqueue_images(date_str)
        print(f"Enqueued {count} images for download")

def start_download_worker():
    print("Starting download worker...")
    worker = SpawnWorker([download_queue], connection=redis_conn)
    worker.work()

def start_processing_worker():
    print("Starting processing worker...")
    worker = Worker([processing_queue], connection=redis_conn)
    worker.work()

def start_insertion_worker():
    print("Starting insertion worker...")
    worker = Worker([insertion_queue], connection=redis_conn)
    worker.work()

def start_all_workers():
    print("Starting all workers...")
    worker = Worker([download_queue, processing_queue, insertion_queue], connection=redis_conn)
    worker.work()

def show_queue_status():
    try:
        download_count = len(download_queue)
        processing_count = len(processing_queue)
        insertion_count = len(insertion_queue)
        
        print(f"Queue Status:")
        print(f"  Download Queue: {download_count} jobs")
        print(f"  Processing Queue: {processing_count} jobs")
        print(f"  Insertion Queue: {insertion_count} jobs")
        
        stats = get_collection_stats()
        if 'error' not in stats:
            print(f"  ChromaDB Collection: {stats['document_count']} documents")
        else:
            print(f"  ChromaDB Error: {stats['error']}")
            
    except Exception as e:
        print(f"Error checking queue status: {e}")

def clear_all_queues():
    download_queue.empty()
    processing_queue.empty()
    insertion_queue.empty()
    print("All queues cleared")

def main():
    parser = argparse.ArgumentParser(description='Weather Camera Image Processing Pipeline')
    parser.add_argument('command', choices=[
        'scrape', 'download-worker', 'processing-worker', 
        'insertion-worker', 'all-workers', 'status', 'clear'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'scrape':
            start_scraper()
        elif args.command == 'download-worker':
            start_download_worker()
        elif args.command == 'processing-worker':
            start_processing_worker()
        elif args.command == 'insertion-worker':
            start_insertion_worker()
        elif args.command == 'all-workers':
            start_all_workers()
        elif args.command == 'status':
            show_queue_status()
        elif args.command == 'clear':
            clear_all_queues()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()