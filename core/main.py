import argparse
from datetime import datetime
import sys
import os
import pandas as pd
from worker_scraper import scrape_and_download_images

# Fix for macOS forking issue
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

def start_scraper():
    print("Starting image scraping and download process...")
    startDate = "2025/01/28"
    endDate = "2025/06/30"  # Use 2025/09/01 date for scraping
    for date in pd.date_range(startDate, endDate):
        date_str = date.strftime("%Y/%m/%d")
        print(f"Scraping images for date: {date_str}")
        count = scrape_and_download_images(date_str)
        print(f"Enqueued {count} images for download")

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
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()