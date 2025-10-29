import json
import logging
import sys
import os
import utils
import boto3
import re
import info
import contextlib

from strands import Agent
from strands.models import BedrockModel
from botocore.config import Config
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("agent")

index = 0
def add_notification(containers, message):
    global index
    if containers is not None:
        containers['notification'][index].info(message)
    index += 1

def add_response(containers, message):
    global index
    containers['notification'][index].markdown(message)
    index += 1
    
status_msg = []
def get_status_msg(status):
    global status_msg
    status_msg.append(status)

    if status != "end)":
        status = " -> ".join(status_msg)
        return "[status]\n" + status + "..."
    else: 
        status = " -> ".join(status_msg)
        return "[status]\n" + status

aws_region = utils.bedrock_region

def get_model(model_name):
    STOP_SEQUENCE = "\n\nHuman:" 
    maxOutputTokens = 4096 # 4k

    models = info.get_model_info(model_name)
    model_info = models[0]
    model_id = model_info['model_id']
    
    # Use region from config.json
    bedrock_region = utils.bedrock_region

    logger.info(f"Using model_id: {model_id}, region: {bedrock_region} (from config.json)")

    # Bedrock client configuration
    bedrock_config = Config(
        read_timeout=900,
        connect_timeout=900,
        retries=dict(max_attempts=3, mode="adaptive"),
    )
    
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=bedrock_region,
        config=bedrock_config
    )

    model = BedrockModel(
        client=bedrock_client,
        model_id=model_id,
        region_name=bedrock_region,  # Explicitly set region for BedrockModel
        max_tokens=maxOutputTokens,
        stop_sequences = [STOP_SEQUENCE],
        temperature = 0.1,
        top_p = 0.9,
        additional_request_fields={
            "thinking": {
                "type": "disabled"
            }
        }
    )
    return model

def load_mcp_config():
    config = None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "mcp.json")
    
    # Debug: print the actual path being used
    logger.info(f"script_dir: {script_dir}")
    logger.info(f"config_path: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False

# Global variables
conversation_manager = SlidingWindowConversationManager(
    window_size=10,  
)
agent = None
knowledge_base_client = repl_coder_client = notion_client = None

def initialize_agent(model_name, mcp_servers):
    """Initialize the global agent with MCP client"""
    # MCP server mapping
    mcp_server_mapping = {
        "RAG": "knowledge_base",
        "Notion": "notionApi", 
        "Code Interpreter": "repl_coder"
    }
    
    # Create clients based on mcp_servers
    active_clients = []
    knowledge_base_client = None
    repl_coder_client = None
    notion_client = None
    
    # Set default servers if none specified
    servers_to_use = mcp_servers if mcp_servers else ["RAG"]
    
    for server_name in servers_to_use:
        if server_name in mcp_server_mapping:
            client_name = mcp_server_mapping[server_name]
            client = create_mcp_client(client_name)
            active_clients.append(client)
            
            # Assign to each client variable
            if server_name == "RAG":
                knowledge_base_client = client
            elif server_name == "Notion":
                notion_client = client
            elif server_name == "Code Interpreter":
                repl_coder_client = client
                
            logger.info(f"Created MCP client for {server_name} -> {client_name}")
        else:
            logger.warning(f"Unknown MCP server: {server_name}")
    
    if not active_clients:
        logger.warning("No valid MCP clients created")
        return None, None, None, None, []
        
    # Create agent within MCP client context manager
    with contextlib.ExitStack() as stack:
        for client in active_clients:
            stack.enter_context(client)
        
        mcp_tools = []
        for client in active_clients:
            mcp_tools.extend(client.list_tools_sync())
        
        logger.info(f"mcp_tools: {mcp_tools}")
        
        tools = []
        tools.extend(mcp_tools)

        tool_list = get_tool_list(tools)
        logger.info(f"tools loaded: {tool_list}")
    
        system_prompt = (
            "당신의 이름은 지민이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
        model = get_model(model_name)

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            conversation_manager=conversation_manager
        )
    
    return agent, knowledge_base_client, repl_coder_client, notion_client, tool_list

def get_tool_info(tool_name, tool_content):
    tool_references = []    
    urls = []
    content = ""

    try:
        if isinstance(tool_content, dict):
            json_data = tool_content
        elif isinstance(tool_content, list):
            json_data = tool_content
        else:
            json_data = json.loads(tool_content)
        
        logger.info(f"json_data: {json_data}")
        if isinstance(json_data, dict) and "path" in json_data:  # path
            path = json_data["path"]
            if isinstance(path, list):
                for url in path:
                    if url and url.strip():  # Only add if not empty string
                        urls.append(url)
            else:
                if path and path.strip():  # Only add if not empty string
                    urls.append(path)            

        for item in json_data:
            logger.info(f"item: {item}")
            if "reference" in item and "contents" in item:
                url = item["reference"]["url"]
                title = item["reference"]["title"]
                content_text = item["contents"][:200] + "..." if len(item["contents"]) > 200 else item["contents"]
                content_text = content_text.replace("\n", "")
                tool_references.append({
                    "url": url,
                    "title": title,
                    "content": content_text
                })
        logger.info(f"tool_references: {tool_references}")

    except json.JSONDecodeError:
        pass

    return content, urls, tool_references

def get_reference(references):
    ref = ""
    if references:
        ref = "\n\n### Reference\n"
        for i, reference in enumerate(references):
            ref += f"{i+1}. [{reference['title']}]({reference['url']}), {reference['content']}...\n"        
    return ref

def filter_mcp_parameters(tool_name, input_params):
    """Filter out unexpected parameters for MCP tools"""
    if not isinstance(input_params, dict):
        return input_params
    
    # Known problematic parameters that should be filtered out
    problematic_params = ['mcp-session-id', 'session-id', 'session_id']
    
    filtered_params = {}
    for key, value in input_params.items():
        if key not in problematic_params:
            filtered_params[key] = value
        else:
            logger.info(f"Filtered out problematic parameter '{key}' for tool '{tool_name}'")
    
    return filtered_params

async def show_streams(agent_stream, containers):
    tool_name = ""
    result = ""
    current_response = ""
    references = []
    image_url = []  

    async for event in agent_stream:
        # logger.info(f"event: {event}")
        if "message" in event:
            message = event["message"]
            logger.info(f"message: {message}")

            for content in message["content"]:      
                logger.info(f"content: {content}")          
                if "text" in content:
                    logger.info(f"text: {content['text']}")

                    result = content['text']
                    current_response = ""

                if "toolUse" in content:
                    tool_use = content["toolUse"]
                    logger.info(f"tool_use: {tool_use}")
                    
                    tool_name = tool_use["name"]
                    input_params = tool_use["input"]
                    
                    # Filter out problematic parameters
                    filtered_input = filter_mcp_parameters(tool_name, input_params)
                    
                    logger.info(f"tool_name: {tool_name}, original_arg: {input_params}, filtered_arg: {filtered_input}")
                                
                refs = []
                if "toolResult" in content:
                    tool_result = content["toolResult"]
                    logger.info(f"tool_name: {tool_name}")
                    logger.info(f"tool_result: {tool_result}")
                    if "content" in tool_result:
                        tool_content = tool_result['content']
                        for content in tool_content:
                            if "text" in content:
                                content, urls, refs = get_tool_info(tool_name, content['text'])
                                logger.info(f"content: {content}")
                                logger.info(f"urls: {urls}")
                                logger.info(f"refs: {refs}")

                                if refs:
                                    for r in refs:
                                        references.append(r)
                                        logger.info(f"refs: {refs}")

                                if urls:
                                    valid_urls = [url for url in urls if url and url.strip()]
                                    if valid_urls:
                                        for url in valid_urls:
                                            image_url.append(url)
                                        logger.info(f"valid_urls: {valid_urls}")
                                    else:
                                        logger.info("유효한 URL이 없습니다.")
                                else:
                                    logger.info("URLs가 비어있습니다.")                                

        if "data" in event:
            text_data = event["data"]
            current_response += text_data

            if containers is not None:
                containers["notification"][index].markdown(current_response)
            continue
        
    # get reference
    # result += get_reference(references)
    
    return result, image_url

def get_tool_list(tools):
    tool_list = []
    for tool in tools:
        if hasattr(tool, 'tool_name'):  # MCP tool
            tool_list.append(tool.tool_name)
        elif hasattr(tool, 'name'):  # MCP tool with name attribute
            tool_list.append(tool.name)
        elif hasattr(tool, '__name__'):  # Function or module
            tool_list.append(tool.__name__)
        elif str(tool).startswith("<module 'strands_tools."):   
            module_name = str(tool).split("'")[1].split('.')[-1]
            tool_list.append(module_name)
        else:
            # For MCP tools that might have different structure
            tool_str = str(tool)
            if 'MCPAgentTool' in tool_str:
                # Try to extract tool name from MCP tool
                try:
                    if hasattr(tool, 'tool'):
                        tool_list.append(tool.tool.name)
                    else:
                        tool_list.append(f"MCP_Tool_{len(tool_list)}")
                except:
                    tool_list.append(f"MCP_Tool_{len(tool_list)}")
            else:
                tool_list.append(str(tool))
    return tool_list

def create_mcp_client(mcp_server_name: str):
    config = load_mcp_config()
    mcp_servers = config["mcpServers"]
    
    mcp_client = None
    for server_name, server_config in mcp_servers.items():
        logger.info(f"server_name: {server_name}")
        logger.info(f"server_config: {server_config}")   

        env = server_config["env"] if "env" in server_config else None

        if server_name == mcp_server_name:
            mcp_client = MCPClient(lambda: stdio_client(
                StdioServerParameters(
                    command=server_config["command"], 
                    args=server_config["args"], 
                    env=env
                )
            ))
            break
    
    return mcp_client

tool_list = None
# Store previous mcp_servers to detect changes
previous_mcp_servers = None

async def initiate(model_name, mcp_servers):
    global agent, knowledge_base_client, repl_coder_client, notion_client, tool_list, previous_mcp_servers
    
    # Check if mcp_servers has changed or agent doesn't exist
    if agent is None or previous_mcp_servers != mcp_servers:
        logger.info(f"MCP servers changed from {previous_mcp_servers} to {mcp_servers}, reinitializing agent")
        agent, knowledge_base_client, repl_coder_client, notion_client, tool_list = initialize_agent(model_name, mcp_servers)
        previous_mcp_servers = mcp_servers.copy() if mcp_servers else []
    
    # Include only active clients in context manager
    active_clients = []
    if knowledge_base_client:
        active_clients.append(knowledge_base_client)
    if repl_coder_client:
        active_clients.append(repl_coder_client)
    if notion_client:
        active_clients.append(notion_client)

    return agent, active_clients