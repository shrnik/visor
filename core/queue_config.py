import os
from rq import Queue, Worker
import redis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=int(os.getenv('REDIS_DB', 0)),
    password=os.getenv('REDIS_PASSWORD', None)
)

test_redis = redis_conn.ping()
if not test_redis:
    raise Exception("Could not connect to Redis. Please ensure Redis server is running.")

download_queue = Queue('download_queue', connection=redis_conn)
processing_queue = Queue('processing_queue', connection=redis_conn)
insertion_queue = Queue('insertion_queue', connection=redis_conn)

def get_worker(queue_names):
    return Worker(queue_names, connection=redis_conn)