import logging
import sys
import json
import boto3
import os

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("utils")

def load_config():
    config = None

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "config.json")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config

config = load_config()

bedrock_region = config['region']
projectName = config['projectName']
knowledge_base_id = config['knowledge_base_id']

