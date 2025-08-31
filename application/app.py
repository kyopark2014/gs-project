import streamlit as st 
import logging
import sys
import os
import chat
import asyncio
import multi_mcp_agent

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

os.environ["DEV"] = "true"  # Skip user confirmation of get_user_input

# title
st.set_page_config(page_title='GS', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "GS Agent": [
        "MCP를 도구로 활용하는 Agent를 이용합니다."
    ],
    "이미지 분석": [
        "이미지를 선택하여 멀티모달을 이용하여 분석합니다."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Strands와 MCP를 이용하여 똑똑한 Agent를 구현합니다." 
        "상세한 코드는 [Github](https://github.com/kyopark2014/gs-project)을 참조하세요."
    )

    st.subheader("🐱 대화 형태")

    # radio selection
    mode = st.radio(
        label="원하는 대화 형태를 선택하세요. ",options=["GS Agent", "이미지 분석"], index=0
    )   
    st.info(mode_descriptions[mode][0])

    mcp_options = [
        "RAG", "Notion", "Code Interpreter"
    ]
    mcp_selections = {}
    default_selections = ["RAG"]

    with st.expander("MCP 옵션 선택", expanded=True):  
        for option in mcp_options:
            default_value = option in default_selections
            mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
    
    # model selection box
    modelName = st.selectbox(
        '🖊️ 사용 모델을 선택하세요',
        ('Claude 4 Sonnet', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet'), index=1
    )    

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'

    uploaded_file = None
    if mode=='이미지 분석':
        st.subheader("🌇 이미지 업로드")
        uploaded_file = st.file_uploader("이미지 분석을 위한 파일을 선택합니다.", type=["png", "jpg", "jpeg"])

    mcpServers = [server for server, is_selected in mcp_selections.items() if is_selected]
    logger.info(f"mcpServers: {mcpServers}")

    chat.update(modelName, debugMode, mcpServers)

    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("대화 초기화", key="clear")

st.title('🔮 '+ mode)

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    
    st.session_state.greetings = False
    st.rerun()  

# Preview the uploaded image in the sidebar
file_name = ""
file_bytes = None
state_of_code_interpreter = False
if uploaded_file is not None and clear_button==False:
    logger.info(f"uploaded_file.name: {uploaded_file.name}")

    if uploaded_file and clear_button==False and mode == '이미지 분석':
        st.image(uploaded_file, caption="이미지 미리보기", use_container_width=True)

        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages() -> None:
    """Print message history
    @returns None
    """
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message and message["images"]:
                for j, url in enumerate(message["images"]):
                    file_name = url[url.rfind('/')+1:] if '/' in url else url
                    st.image(url, caption=file_name, use_container_width=True)
                    
display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "아마존 베드락을 이용하여 주셔서 감사합니다. Agent를 이용해 향상된 대화를 즐기실 수 있습니다."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    uploaded_file = None
    
    st.session_state.greetings = False
    chat.initiate()
    st.rerun()    

# Always show the chat input
if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")

    with st.chat_message("assistant"):        
        sessionState = ""            
        
        with st.status("thinking...", expanded=True, state="running") as status:     
            containers = {
                "tools": st.empty(),
                "status": st.empty(),
                "notification": [st.empty() for _ in range(500)]
            }  
                   
            image_url = None
            if mode == "GS Agent":
                response = asyncio.run(multi_mcp_agent.run_agent(query=prompt, containers=containers))
            elif mode == "이미지 분석":
                if uploaded_file is None or uploaded_file == "":
                    st.error("파일을 먼저 업로드하세요.")
                    st.stop()

                else:
                    summary = chat.summarize_image(file_bytes, prompt, st)
                    st.write(summary)

                    st.session_state.messages.append({"role": "assistant", "content": summary})
                    response = summary

            assistant_message = {
                "role": "assistant", 
                "content": response,
                "images": image_url if image_url else []
            }
            st.session_state.messages.append(assistant_message)
            
            if image_url:
                for url in image_url:
                    logger.info(f"url: {url}")
                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)