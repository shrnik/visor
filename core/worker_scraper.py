import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta
import time
from PIL import Image
from core.queue_config import download_queue, processing_queue
import re

BASE_URL = "https://metobs.ssec.wisc.edu/pub/cache/aoss/cameras/east/img/"

def get_day_url(date: datetime) -> str:
    date_str = date.strftime("%Y/%m/%d")
    print(f"Fetching images for date: {date_str}")
    return urljoin(BASE_URL, f"{date_str}/orig/")

def extract_timestamp_from_url(url) -> datetime:
    timestamp_match = re.match(r'(\d{2})_(\d{2})_(\d{2})\.trig\+00', url)
    if timestamp_match:
        return datetime.strptime(timestamp_match.group(0), '%y_%m_%d')
    return None

def scrape_image_links(date: datetime,):
    try:
        base_url = get_day_url(date)
        response = requests.get(base_url, timeout=120)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        image_links = []
        
        for link in soup.find_all('a'):
            print(link)
            href = link.get('href')
            if href and (href.endswith('.jpg') or href.endswith('.jpeg') or href.endswith('.png')):
                image_pattern = r'(\d{2}_\d{2}_\d{2}\.trig\+00\.jpg)'
                if not re.match(image_pattern, href):
                    continue
                full_url = urljoin(base_url, href)
                timestamp = extract_timestamp_from_url(full_url)
                image_links.append({
                    'url': full_url,
                    'filename': href,
                    'date': date.strftime("%Y/%m/%d"),
                    'timestamp': timestamp
                })
        # take every 6th image
        image_links = image_links[::6]
        # save links to a file
        with open(f"image_links_{date.strftime('%Y_%m_%d')}.txt", 'w') as f:
            for link in image_links:
                f.write(f"{link['url']}\n")
        print(f"Found {len(image_links)} images")
        return image_links
    except Exception as e:
        print(f"Error scraping {base_url}: {e}")
        return []

def download_image(image_data):
    try:
        url = image_data['url']
        filename = image_data['filename']
        date = image_data['date']
        
        os.makedirs('downloaded_images', exist_ok=True)
        # folder structure: downloaded_images/YYYY_MM_DD/
        date_folder = os.path.join('downloaded_images', date.replace('/', '_'))
        os.makedirs(date_folder, exist_ok=True)
        local_path = os.path.join(date_folder, filename)

        print("checking")
        if os.path.exists(local_path):
            print(f"Image {filename} already exists, skipping download")
            return {
                'success': True,
                'local_path': local_path,
                'url': url,
                'filename': filename,
                'already_existed': True
            }
        print(f"calling image {filename} from {url}")
        
        try :

            response = requests.get(url, timeout=60)
            response.raise_for_status()
            print(f"writing image {filename} from {url}")
            with open(local_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Error downloading image {filename} from {url}: {e}")
            raise e
        try:
            with Image.open(local_path) as img:
                img.verify()
        except Exception as e:
            os.remove(local_path)
            raise Exception(f"Downloaded file is not a valid image: {e}")
        print (f"Downloaded image {filename} to {local_path}")
        return {
            'success': True,
            'local_path': local_path,
            'url': url,
            'filename': filename,
            'already_existed': False
        }
        
    except Exception as e:
        print(f"Error downloading {image_data.get('url', 'unknown')}: {e}")
        return {
            'success': False,
            'error': str(e),
            'url': image_data.get('url', 'unknown')
        }

def scrape_and_enqueue_images(date):
    print("Starting image scraping...")
    image_links = scrape_image_links(date)
    
    print(f"Found {len(image_links)} images to process")
    
    for image_data in image_links:
        job = download_queue.enqueue(download_and_process_image, image_data, timeout=300)
        print(f"Enqueued download job {job.id} for {image_data['filename']}")
    
    return len(image_links)

def download_and_process_image(image_data, timeout=300):
    print(f"Downloading image {image_data['filename']} from {image_data['url']}")
    try: 
        result = download_image(image_data)
        
        if result['success']:
            processing_job = processing_queue.enqueue(
                'core.worker_processor.process_image_embedding',
                result,
                timeout=600
            )
            print(f"Image {result['filename']} downloaded, enqueued for processing: {processing_job.id}")
            return result
        else:
            print(f"Failed to download {image_data.get('filename', 'unknown')}: {result.get('error')}")
            return result
    except Exception as e:
        print(f"Error in download_and_process_image for {image_data.get('filename', 'unknown')}: {e}")
        raise e

if __name__ == "__main__":
    scrape_and_enqueue_images()