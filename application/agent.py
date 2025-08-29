import json
import logging
import sys
import os
import utils
import boto3
import re
import chat

from typing import Dict, List, Optional
from strands import Agent
from strands.models import BedrockModel
from botocore.config import Config
from strands_tools import memory, retrieve
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

model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
aws_region = utils.bedrock_region

def get_model():
    STOP_SEQUENCE = "\n\nHuman:" 
    maxOutputTokens = 4096 # 4k

    # Bedrock client configuration
    bedrock_config = Config(
        read_timeout=900,
        connect_timeout=900,
        retries=dict(max_attempts=3, mode="adaptive"),
    )
    
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=aws_region,
        config=bedrock_config
    )

    model = BedrockModel(
        client=bedrock_client,
        model_id=chat.model_id,
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
mcp_client = None

def initialize_agent():
    """Initialize the global agent with MCP client"""
    mcp_client = create_mcp_client("knowledge_base")
        
    # Create agent within MCP client context manager
    with mcp_client as client:
        mcp_tools = client.list_tools_sync()
        logger.info(f"mcp_tools: {mcp_tools}")
        
        tools = []
        tools.extend(mcp_tools)

        tool_list = get_tool_list(tools)
        logger.info(f"tools loaded: {tool_list}")
    
        system_prompt = (
            "당신의 이름은 현민이고, 질문에 대해 친절하게 답변하는 사려깊은 인공지능 도우미입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다." 
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
        )
        model = get_model()

        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            conversation_manager=conversation_manager
        )
    
    return agent, mcp_client, tool_list

def create_filtered_mcp_tools(client):
    """Create MCP tools with parameter filtering"""
    
    original_tools = client.list_tools_sync()
    filtered_tools = []
    
    for tool in original_tools:
        if hasattr(tool, 'tool') and hasattr(tool.tool, 'name'):
            # Create a wrapper that filters parameters
            original_call = tool.call_async
            
            async def filtered_call(tool_use, invocation_state):
                # Filter out problematic parameters
                if hasattr(tool_use, 'input') and isinstance(tool_use.input, dict):
                    filtered_input = filter_mcp_parameters(tool.tool.name, tool_use.input)
                    # Create a new tool_use with filtered input
                    tool_use.input = filtered_input
                
                return await original_call(tool_use, invocation_state)
            
            # Replace the call method
            tool.call_async = filtered_call
            filtered_tools.append(tool)
        else:
            filtered_tools.append(tool)
    
    return filtered_tools

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
                    urls.append(url)
            else:
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

    async for event in agent_stream:
        # logger.info(f"event: {event}")
        if "message" in event:
            message = event["message"]
            logger.info(f"message: {message}")

            for content in message["content"]:      
                logger.info(f"content: {content}")          
                if "text" in content:
                    logger.info(f"text: {content['text']}")

                    if containers is not None:
                        add_response(containers, content['text'])

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
                    
                    if containers is not None:       
                        add_notification(containers, f"tool name: {tool_name}, arg: {filtered_input}")
                        containers['status'].info(get_status_msg(f"{tool_name}"))
            
                refs = []
                if "toolResult" in content:
                    tool_result = content["toolResult"]
                    logger.info(f"tool_name: {tool_name}")
                    logger.info(f"tool_result: {tool_result}")
                    if "content" in tool_result:
                        tool_content = tool_result['content']
                        for content in tool_content:
                            if "text" in content:
                                if containers is not None:
                                    add_notification(containers, f"tool result: {content['text']}")

                                content, urls, refs = get_tool_info(tool_name, content['text'])
                                logger.info(f"content: {content}")
                                logger.info(f"urls: {urls}")
                                logger.info(f"refs: {refs}")

                                if refs:
                                    for r in refs:
                                        references.append(r)
                                        logger.info(f"refs: {refs}")

        if "data" in event:
            text_data = event["data"]
            current_response += text_data

            if containers is not None:
                containers["notification"][index].markdown(current_response)
            continue
        
    # get reference
    # result += get_reference(references)
    
    return result

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
async def run_agent(query: str, containers):
    global index, status_msg
    global agent, mcp_client, tool_list
    index = 0
    status_msg = []
    
    containers['status'].info(get_status_msg(f"(start"))  

    # Initialize agent if not exists
    if agent is None:
        agent, mcp_client, tool_list = initialize_agent()

    if chat.debug_mode and containers is not None and tool_list:
        containers['tools'].info(f"tool_list: {tool_list}")
    
    with mcp_client as client:
        agent_stream = agent.stream_async(query)
        result = await show_streams(agent_stream, containers)

    logger.info(f"result: {result}")

    containers['status'].info(get_status_msg(f"end)"))

    return result
