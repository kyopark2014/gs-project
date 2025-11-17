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

script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "config.json")
    
def load_config():
    config = {}

    try:    
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)    
    except Exception as e:
        logger.error(f"Error loading config: {e}")

        projectName = 'gs-project'
        knowledge_base_id = 'GT6MDKEVWG'
        user_id = 'gs_agent'

        config['projectName'] = projectName
        config['knowledge_base_id'] = knowledge_base_id
        config['user_id'] = user_id

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        logger.info(f"config was saved to {config_path}")
        pass
        
    return config

config = load_config()

bedrock_region = config.get('region', 'us-west-2')


