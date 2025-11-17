# Agent와 Notion MCP

여기서는 Strands SDK 기반의 chatbot을 이용해 RAG (Knowledge Base), Notion, Code Interpreter를 MCP를 활용하는 것을 설명합니다. 전체적인 architecture는 아래와 같습니다. Knowledge Base를 이용해 사내의 중요한 문서를 열람하고, Notion으로 정리된 각종 메뉴얼을 참고합니다. 또한, Code Interpreter를 이용해 복잡한 데이터를 분석하고 필요시 다이어그램을 그래서 이해를 돕습니다. 여기서 생성된 agent는 streamlit을 이용해 UI를 제공하고 ALB와 CloudFront를 이용하여 안전하게 활용할 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/7adb7458-185c-4b38-b9bd-e3639eaa6c9d" />

## 주요 구현

### Strands Agent

[agent.py](./application/agent.py)와 같이 app에서 agent를 실행하면 아래와 같이 run_agent가 실행됩니다. 이때 최초 실행이 되면 아래와 같이 initialize_agent()로 agent를 생성합니다. mcp_client가 준비가 되면 아래와 같이 agent를 [stream_async](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/streaming/async-iterators/)을 이용해 실행됩니다. Strands agent는 하나의 입력을 multi-step reasoning을 통해 답을 찾아가므로 중간의 출력들을 아래와 같이 show_streams으로 보여줍니다. 

```python
async def run_agent(query: str, containers):
    active_clients = []
    if knowledge_base_client:
        active_clients.append(knowledge_base_client)
    if repl_coder_client:
        active_clients.append(repl_coder_client)
    if notion_client:
        active_clients.append(notion_client)
    
    with contextlib.ExitStack() as stack:
        for client in active_clients:
            stack.enter_context(client)        
        agent_stream = agent.stream_async(query)
        result, image_url = await show_streams(agent_stream, containers)
    return result, image_url
```

[mcp.json](./application/mcp.json)에서는 MCP 서버에 대한 정보를 가지고 있습니다. 이 정보를 이용하여 MCPClient를 생성할 수 있습니다. mcp.json의 MCP 서버의 정보인 command, args, env를 이용해 [StdioServerParameters](https://github.com/strands-agents/sdk-python?tab=readme-ov-file#mcp-support)를 구성합니다.

```python
def create_mcp_client(mcp_server_name: str):
    config = load_mcp_config()
    mcp_servers = config["mcpServers"]
    
    mcp_client = None
    for server_name, server_config in mcp_servers.items():
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
```

아래와 같이 "knowledge_base"를 사용하는 MCP agent를 아래와 같이 create_mcp_client로 생성합니다. 또한 [list_tools_sync](https://github.com/strands-agents/sdk-python?tab=readme-ov-file#mcp-support)를 이용해 tool에 대한 정보를 가져와서 tools에 추가합니다. 이후 아래와 같이 agent를 생성합니다.

```python
def initialize_agent():
    """Initialize the global agent with MCP client"""
    mcp_client = create_mcp_client("knowledge_base")
        
    # Create agent within MCP client context manager
    with mcp_client as client:
        mcp_tools = client.list_tools_sync()        
        tools = []
        tools.extend(mcp_tools)
        tool_list = get_tool_list(tools)    
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
```

여러개의 MCP 서버를 사용할 경우에는 [multi_mcp_agent.py](./application/multi_mcp_agent.py)을 참조합니다. 여기에서는 아래와 같이 knowledge_base_mcp_client와 repl_coder_client를 이용해 mcp client를 생성하고 mcp_tools를 extend해서 활용하여야 합니다. 

```python
def initialize_agent():
    """Initialize the global agent with MCP client"""
    knowledge_base_mcp_client = create_mcp_client("knowledge_base")
    repl_coder_client = create_mcp_client("repl_coder")
        
    # Create agent within MCP client context manager
    with knowledge_base_mcp_client, repl_coder_client:
        mcp_tools = knowledge_base_mcp_client.list_tools_sync()
        mcp_tools.extend(repl_coder_client.list_tools_sync())
```        

### MCP Servers

[mcp_retrieve.py]에서는 Knowledge Base로부터 관련된 문서를 조회합니다. bedrock-agent-runtime로 client를 정의하고 retrieve를 이용해 질문과 관련된 문서를 조회합니다. 얻어진 문서에서 text와 url과 같은 정보를 추출합니다.

```python
bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime", region_name=bedrock_region)

def retrieve(query):
    response = bedrock_agent_runtime_client.retrieve(
        retrievalQuery={"text": query},
        knowledgeBaseId=knowledge_base_id,
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": number_of_results},
            },
        )
    retrieval_results = response.get("retrievalResults", [])
    json_docs = []
    for result in retrieval_results:
        text = url = name = None
        if "content" in result:
            content = result["content"]
            if "text" in content:
                text = content["text"]

        if "location" in result:
            location = result["location"]
            if "s3Location" in location:
                uri = location["s3Location"]["uri"] if location["s3Location"]["uri"] is not None else ""
                
                name = uri.split("/")[-1]
                url = uri # TODO: add path and doc_prefix
                
            elif "webLocation" in location:
                url = location["webLocation"]["url"] if location["webLocation"]["url"] is not None else ""
                name = "WEB"

        json_docs.append({
            "contents": text,              
            "reference": {
                "url": url,                   
                "title": name,
                "from": "RAG"
            }
        })
```

[mcp_server_retrieve.py](./application/mcp_server_retrieve.py)에서는 아래와 같이 FastMCP를 이용해 Knowledge Base를 조회하는 MCP 기능을 구현합니다. 이때 retrieve라는 tool을 아래와 같이 정의할 수 있고 agent은 docstring을 참조하여 적절한 tool을 선택합니다. 


```python
from mcp.server.fastmcp import FastMCP 

mcp = FastMCP(
    name = "mcp-retrieve",
    instructions=(
        "You are a helpful assistant. "
        "You retrieve documents in RAG."
    ),
)

@mcp.tool()
def retrieve(keyword: str) -> str:
    """
    Query the keyword using RAG based on the knowledge base.
    keyword: the keyword to query
    return: the result of query
    """
    return mcp_retrieve.retrieve(keyword)

if __name__ =="__main__":
    print(f"###### main ######")
    mcp.run(transport="stdio")
```




## 설치 및 활용

### RAG 구현

[Knowledge Base](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/knowledge-bases)에 접속해서 [Create]를 선택하여 RAG를 생성합니다. 완료가 되면 Knowledge Base의 ID를 확인합니다.

Amazon S3에 아래와 같이 파일을 업로드합니다. 

<img width="400" alt="noname" src="https://github.com/user-attachments/assets/42f530cf-11eb-456f-be5c-0ca58fe35fc7" />

[Knowledge Base Console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/knowledge-bases)에 접속해서 생성한 Knowledge Bases를 선택한 후에 아래와 같이 sync를 선택합니다. Sync가 완료가 되면 [Test Knowledge Base]를 선택하여 정상적으로 문서 정보를 가져오는지 확인합니다. 

<img width="600" alt="noname" src="https://github.com/user-attachments/assets/efd6aa45-2bc4-43b4-8fcb-d53252c09cce" />

### Notion MCP

Notion에서는 [Official Notion MCP Server](https://github.com/makenotion/notion-mcp-server)와 같은 MCP 서버를 제공하고 있습니다. 아래 방식으로 token을 생성한 후에, mcp.json에 관련정보를 설정한 후 agent에서 활용합니다.

1) [Notion API Integration](https://www.notion.so/profile/integrations)에 접속하여 [새 API 통합]을 선택합니다.
2) 아래와 같이 입력후 저장합니다.

<img width="573" height="535" alt="image" src="https://github.com/user-attachments/assets/787561ad-0b61-4a1a-91ea-72d7556e6358" />

3) API Secret를 복사합니다. 

4) 사용권한 Tab에서 적절한 페이지를 선택합니다.

<img width="664" height="340" alt="image" src="https://github.com/user-attachments/assets/872e7054-1135-4dde-aa6a-756a7ad928b0" />


### Strands Agent의 활용

Strands agent는 multi-step reasoning을 통해 향상된 RAG 검색을 가능하게 해줍니다. 이를 활용하기 위해 먼저 아래와 같이 git으로 부터 소스를 가져옵니다.

```text
git clone https://github.com/kyopark2014/GS-project
```

"application" 폴더의 [config.json](./application/config.json)을 선택한 후에 아래와 같이 knowledge_base_id를 업데이트 합니다. knowledge_base_id은 생성한 Knowledge Base의 ID입니다.

```java
{
    "projectName":"GS-project",
    "region":"us-west-2",
    "knowledge_base_id":"O2IGZXMQXO"
 }
```

이제 필요한 패키지를 설치합니다.

```text
pip install streamlit streamlit-chat boto3 asyncio strands-agents strands-agents-tools mcp langchain_experimental graphviz matplotlib
```

[application/mcp.json.sample](./application/mcp.json.sample) 파일을 아래와 같이 복사해서 mcp.json 파일을 생성합니다.

```text
cp application/mcp.json.sample application/mcp.json
```

이제 mcp.json 파일을 열어서 아래의 NOTION_TOKEN를 업데이트 합니다.

```java
{
    "mcpServers": {
        "knowledge_base": {
            "command": "python",
            "args": ["application/mcp_server_retrieve.py"]
        },
        "repl_coder": {
            "command": "python",
            "args": ["application/mcp_server_repl_coder.py"]
        },
        "notionApi": {
            "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "env": {
                "NOTION_TOKEN": "ntn_Token 입력이 필요합니다."
            }
        }
    }
}
```

이후 아래와 같이 streamlit을 실행합니다.

```text
streamlit run application/app.py
```

죄측의 메뉴에서 사용하는 모델을 선택할 수 있으며, "Debug Mode"로 최종 결과와 전체 결과를 구분하여 확인할 수 있습니다. 

<img width="400" alt="image" src="https://github.com/user-attachments/assets/4d72b952-90cc-4d1e-9f67-a200a2d01578" />

### 실행 결과

MCP를 이용하면 Notion 문서를 손쉽게 조회할 수 있습니다. 먼저 Notion에 아래와 같은 AgentCore에 대한 설명 자료를 추가하였습니다.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/4852e7f6-2e87-4e74-af04-a42557c7ab41" />

이후 아래와 같이 Notion의 문서를 조회할 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/30c85b2b-29a5-4475-b12d-a817965a296c" />

Code Interpreter MCP를 이용해 아래와 같이 그래프를 그릴 수 있습니다.

<img width="700" alt="image" src="https://github.com/user-attachments/assets/f3fad29e-92d2-41e4-9888-651085bbfc17" />


## Production Deployment

### AgentCore Deployment

#### 배포 준비

아래의 명령어로 gs_agent 폴더로 이동한 후에 agent 동작에 필요한 policy를 생성합니다.

```text
cd runtime/gs_agent 
python create_iam_policies.py
```

이후 아래와 같이 agent 인증에 필요한 token을 생성합니다. 생성된 token은 secret에 보관되고 agent 호출시 사용됩니다.

```text
python create_bearer_token.py
```

[mcp.json.sample.json](./runtime/gs_agent/mcp.json.sample.json)을 아래와 같이 이름을 변경하고 파일을 열어서 내용을 업데이트합니다.

```text
mv mcp.json.sample.json mcp.json
```

#### Knowledge Base 정보 업데이트

config.json 파일을 열어서 아래와 같이 knowledge_base_id를 추가합니다.

```java
"knowledge_base_id":"AT1MDKAVWG"
```

#### Docker 이미지 준비

아래와 같이 [build-docker.sh](./runtime/gs_agent/build-docker.sh)를 이용해 docker를 빌드합니다. 이때 PC에 Docker Desktop이 설치되어 있어야 합니다.

```text
./build-docker.sh
```

Local에서 테스트 하기 위하여 아래와 같이 실행합니다.

```text
./run-docker.sh
```


#### Local 동작 시험

별도 터미널을 열어서 아래의 명령어로 실행 결과를 확인합니다.

```text
docker logs gs_gs_agent-container -f
```


아래와 같이 local에서 동작을 테스트 할 수 있습니다.

```text
python test_runtime_local.py
```

#### ECR 배포

local에서 동작에 문제가 없을 경우에 아래와 같이 ECR에 push 합니다. 

```text
./push-to-ecr.sh
```

#### AgentCore 배포

아래 명령어로 AgentCore에 배포합니다.

```text
python create_agent_runtime.py
```

배포가 성공하면 아래와 같이 AgentCore Runtime에서 확인할 수 있습니다.

<img width="729" height="158" alt="image" src="https://github.com/user-attachments/assets/0f4fdf7c-1afe-4ada-bcef-cfef1adc6d60" />

#### AgentCore Runtime 동작확인

아래와 같이 AgentCore Runtime에 배포된 agent를 테스트 할 수 있습니다.

```text
python test_runtime_remote.py
```




### Tips

사용할 수 있는 모델의 확인 방법은 아래와 같습니다.

```text
aws bedrock list-foundation-models --region=ap-northeast-2 --by-provider anthropic --query "modelSummaries[*].modelId"
```

이때의 결과는 아래와 같습니다.

```java
[
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0:200k",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0:28k",
    "anthropic.claude-3-sonnet-20240229-v1:0:200k",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-sonnet-4-20250514-v1:0"
]
```

## Memory 

Chatbot은 연속적인 사용자의 대화를 이용하여 사용자의 경험을 향상시킬 수 있습니다. 일반 대화형 chatbot에서는 이전 대화를 [sliding window](https://langchain-ai.github.io/langgraph/concepts/memory/) 형태로 context에 포함하므로 사용할 수 있는 대화의 숫자가 제한되고, 이전 대화가 필요하지 않는 경우에도 context를 사용하는 문제가 있습니다. 여기에서는 short/long term memory를 지원하는 MCP를 이용하여 생성형 AI 애플리케이션이 필요할 때마다 메모리를 조회하여 활용하는 방법을 설명합니다. 이전 대화 내용은 필요할 때에만 참조되고, 사용자의 프로필과 같은 주요한 정보도 필요에 따라 MCP를 이용해 조회하여 사용할 수 있습니다. 

아래 architecture에는 short/long term meory를 MCP로 활용합니다. [AgentCore memory](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory-getting-started.html)를 이용하면 별도의 DB를 만들어서 관리하지 않아도 생성형 AI 애플리케이션에 필요한 short/long term memory를 손쉽게 활용할 수 있습니다. 대화중 발생하는 transaction은 short-term memory에 저장되며, 주로 최근 n개의 메시지를 가져오는 방식으로 활용됩니다. 대화중 중요한 정보는 long-term memory에 namespace를 이용해 저장됩니다. 저장된 정보를 조회할 때에는 namespace를 이용해 검색의 범위를 조정할 수 있습니다. Long-term memory는 prompt를 가진 strategy을 이용해 사용자의 메시지로부터 필요한 정보를 자동으로 추출합니다. 추출 과정에서는 Amazon Bedrock의 Anthropic의 Claude나 OpenAI의 OSS 같은 모델을 활용할 수 있습니다. 이와 같이 short/long term memory를 지원하는 MCP를 활용하면 대화 중에 필요한 정보를 가져와서 활용할 수 있습니다.

<img width="813" height="372" alt="image" src="https://github.com/user-attachments/assets/00d18ec4-0c26-408b-a89c-694b3ddbecb4" />


### 메모리 저장

아래와 같이 agentcore의 memory에 저장합니다. 상세한 코드는 [agentcore_memory.py](./application/agentcore_memory.py)을 참조합니다.

```python
bedrock_region = "us-west-2"
memory_client = MemoryClient(region_name=bedrock_region)

def save_conversation_to_memory(memory_id, actor_id, session_id, query, result):
    # Truncate text to AWS Bedrock limit (9000 characters)
    max_length = 9000
    truncate_suffix = "... [truncated]"
    suffix_length = len(truncate_suffix)
    max_content_length = max_length - suffix_length  # Reserve space for suffix
    
    query_trimmed = query.strip()
    result_trimmed = result.strip()
    
    if len(query_trimmed) > max_length:
        query_trimmed = query_trimmed[:max_content_length] + truncate_suffix
        # Ensure final length doesn't exceed max_length
        if len(query_trimmed) > max_length:
            query_trimmed = query_trimmed[:max_length]
    
    if len(result_trimmed) > max_length:
        result_trimmed = result_trimmed[:max_content_length] + truncate_suffix
        # Ensure final length doesn't exceed max_length
        if len(result_trimmed) > max_length:
            result_trimmed = result_trimmed[:max_length]

    event_timestamp = datetime.now(timezone.utc)
    conversation = [
        (query_trimmed, "USER"),
        (result_trimmed, "ASSISTANT")
    ]

    memory_result = memory_client.create_event(
        memory_id=memory_id,
        actor_id=actor_id, 
        session_id=session_id, 
        event_timestamp=event_timestamp,
        messages=conversation
    )
```

### Long Term Memory

Long term meory를 위해 필요한 정보에는 memory, actor, session, namespace가 있습니다. 아래와 같이 이미 저장된 값이 있다면 가져오고, 없다면 생성합니다. 상세한 코드는 [langgraph_agent.py](./application/langgraph_agent.py)을 참조합니다.

```python
# initate memory variables
memory_id, actor_id, session_id, namespace = agentcore_memory.load_memory_variables(chat.user_id)
logger.info(f"memory_id: {memory_id}, actor_id: {actor_id}, session_id: {session_id}, namespace: {namespace}")

if memory_id is None:
    # retrieve memory id
    memory_id = agentcore_memory.retrieve_memory_id()
    logger.info(f"memory_id: {memory_id}")        
    
    # create memory if not exists
    if memory_id is None:
        memory_id = agentcore_memory.create_memory(namespace)
    
    # create strategy if not exists
    agentcore_memory.create_strategy_if_not_exists(memory_id=memory_id, namespace=namespace, strategy_name=chat.user_id)

    # save memory variables
    agentcore_memory.update_memory_variables(
        user_id=chat.user_id, 
        memory_id=memory_id, 
        actor_id=actor_id, 
        session_id=session_id, 
        namespace=namespace)
```

생성형 AI 애플리케이션에서는 대화중 필요한 메모리 정보가 있다면 이를 MCP를 이용해 조회합니다. [mcp_server_long_term_memory.py](./application/mcp_server_long_term_memory.py)에서는 long term memory를 이용해 대화 이벤트를 저장하거나 조회할 수 있습니다. 아래는 신규로 레코드를 생성하는 방법입니다.

```python
response = create_event(
    memory_id=memory_id,
    actor_id=actor_id,
    session_id=session_id,
    content=content,
    event_timestamp=datetime.now(timezone.utc),
)
event_data = response.get("event", {}) if isinstance(response, dict) else {}
```

대화에 필요한 정보는 아래와 같이 조회합니다.

```python
contents = []
response = retrieve_memory_records(
    memory_id=memory_id,
    namespace=namespace,
    search_query=query,
    max_results=max_results,
    next_token=next_token,
)
relevant_data = {}
if isinstance(response, dict):
    if "memoryRecordSummaries" in response:
        relevant_data["memoryRecordSummaries"] = response["memoryRecordSummaries"]    
    for memory_record_summary in relevant_data["memoryRecordSummaries"]:
        json_content = memory_record_summary["content"]["text"]
        content = json.loads(json_content)
        contents.append(content)
```

"내가 다니는 회사에 대해 소개해줘"라고 질문하면 long term memory에서 사용자에 대한 정보를 가져와서 아래와 같은 결과를 보여줍니다.

<img width="719" height="721" alt="image" src="https://github.com/user-attachments/assets/da23b49e-5a4b-402f-b314-62211bb59b30" />





## Reference

[Hosting a local MCP server](https://developers.notion.com/docs/hosting-open-source-mcp)

[Official Notion MCP Server](https://github.com/makenotion/notion-mcp-server)

[Common MCP clients](https://developers.notion.com/docs/common-mcp-clients)

[Notion API Integration](https://www.notion.so/profile/integrations)
