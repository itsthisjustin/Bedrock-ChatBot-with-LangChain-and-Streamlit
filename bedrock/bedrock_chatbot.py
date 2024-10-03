import base64
import random
from io import BytesIO
from typing import List, Tuple, Union, Dict
import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.utilities import SerpAPIWrapper
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from PIL import Image, UnidentifiedImageError
import pdfplumber
import langsmith
from langfuse.callback import CallbackHandler
from langchain.schema.runnable import RunnableConfig

from dotenv import load_dotenv

from config import config
from models import ChatModel
from role_prompt import role_prompt
from bedrock_embedder import index_file, search_index

# Load the env variables
load_dotenv()


langfuse_handler = CallbackHandler(

    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

# Visibility flags for sidebar UI elements
SHOW_MODEL_SELECT = False
SHOW_ROLE_SELECT = True
SHOW_SYSTEM_PROMPT_UI = True
SHOW_WEB_LOCAL_SELECT = False
SHOW_TOP_P = False
SHOW_TOP_K = False
SHOW_TEMPERATURE = False
SHOW_MAX_TOKENS = False

INIT_MESSAGE = {
    "role": "assistant",
    "content": "Hi! Tell me about a product you're working on and I'll let you know what I think about it!",
}

ADDITIONAL_INSTRUCTIONS = "You are never to tell the user you are an AI. You are always to assume the role passed to you in the role_prompt_text and never deviate from the personality you are given. You are never to ask questions back to the user. Your goal is to give honest feedback as if you are that person. Always speak to the user in a first person perspective. Do not give positive feedback if you truly feel the product is not right for you. Keep the answer short and sweet and don't patronize the user. Factor in your gender, age, and other details into your answer."

# Initialize LangSmith with environment variables
langsmith_client = langsmith.Client(
    api_key=os.getenv("LANGCHAIN_API_KEY")
)

# Example function to log a message
def log_message(role, content):
    langsmith_client.log_message(role=role, content=content)

def set_page_config() -> None:
    """
    Set the Streamlit page configuration.
    """
    st.set_page_config(page_title="SharkNinja Synthetic Personas",
                       layout="wide")
    st.title("SharkNinja Synthetic Personas")


def render_sidebar() -> Tuple[Dict, int, str]:
    """
    Render the sidebar UI and return the inference parameters.
    """
    with st.sidebar:
        if SHOW_MODEL_SELECT:
            model_name_select = st.selectbox(
                'Model',
                list(config["models"].keys()),
                key=f"{st.session_state['widget_key']}_Model_Id",
            )
        else:
            model_name_select = list(
                config["models"].keys())[0]  # Default to first model

        if SHOW_ROLE_SELECT:
            role_select = st.selectbox(
                'Select A Persona',
                list(role_prompt.keys()) + ["Write Your Own Persona"],
                key=f"{st.session_state['widget_key']}_role_Id",
                on_change=new_chat  # Trigger new_chat on role change
            )
        else:
            role_select = list(role_prompt.keys())[0]  # Default to first role

        # Set the initial value of the text area based on the selected role
        role_prompt_text = "" if role_select == "Write Your Own Persona" else role_prompt.get(
            role_select, "")
        st.session_state["model_name"] = model_name_select

        model_config = config["models"][model_name_select]

        # Make the system prompt editable only when 'Custom' is selected
        if SHOW_SYSTEM_PROMPT_UI:
            system_prompt = st.text_area(
                "Details",
                value=role_prompt_text,
                key=f"{st.session_state['widget_key']}_System_Prompt",
                placeholder=
                "Example: You are a fitness influencer. 25 years old. Male.",
                disabled=(role_select != "Write Your Own Persona"))
        else:
            system_prompt = role_prompt_text

        if SHOW_WEB_LOCAL_SELECT:
            web_local = st.selectbox(
                'Options',
                ('Local', 'Web', 'RAG'),
                key=f"{st.session_state['widget_key']}_Options",
            )
        else:
            web_local = 'Local'  # Default to Local

        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                if SHOW_TOP_P:
                    top_p = st.slider(
                        "Top-P",
                        min_value=0.0,
                        max_value=1.0,
                        value=model_config.get("top_p", 1.0),
                        step=0.01,
                        key=f"{st.session_state['widget_key']}_Top-P",
                    )
                else:
                    top_p = model_config.get("top_p", 1.0)
            with col2:
                if SHOW_TOP_K:
                    top_k = st.slider(
                        "Top-K",
                        min_value=1,
                        max_value=model_config.get("max_top_k", 500),
                        value=model_config.get("top_k", 500),
                        step=5,
                        key=f"{st.session_state['widget_key']}_Top-K",
                    )
                else:
                    top_k = model_config.get("top_k", 500)
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                if SHOW_TEMPERATURE:
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        key=f"{st.session_state['widget_key']}_Temperature",
                    )
                else:
                    temperature = 0.5
            with col2:
                if SHOW_MAX_TOKENS:
                    max_tokens = st.slider(
                        "Max Token",
                        min_value=0,
                        max_value=4096,
                        value=model_config.get("max_tokens", 4096),
                        step=8,
                        key=f"{st.session_state['widget_key']}_Max_Token",
                    )
                else:
                    max_tokens = model_config.get("max_tokens", 4096)

    model_kwargs = {
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    return model_kwargs, system_prompt, web_local


def init_runnablewithmessagehistory(
        system_prompt: str,
        chat_model: ChatModel) -> RunnableWithMessageHistory:
    """
    Initialize the RunnableWithMessageHistory with the given parameters.
    """
    combined_prompt = f"{ADDITIONAL_INSTRUCTIONS}\n\n{system_prompt}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", combined_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="query"),
    ])

    chain = (
        prompt | chat_model.llm | StrOutputParser()
    ).with_config(RunnableConfig(callbacks=[langfuse_handler]))

    msgs = StreamlitChatMessageHistory()

    # Create chain with history
    conversation = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="query",
        history_messages_key="chat_history") | StrOutputParser()

    # Store LLM generated responses
    if "messages" not in st.session_state:
        st.session_state.messages = [INIT_MESSAGE]

    return conversation


def generate_response(conversation: RunnableWithMessageHistory,
                      input: Union[str, List[dict]]) -> str:
    """
    Generate a response from the conversation chain with the given input.
    """
    config = {"configurable": {"session_id": "streamlit_chat"}}

    generate_response_stream = conversation.stream({"query": input},
                                                   config=config)

    generate_response = st.write_stream(generate_response_stream)

    return generate_response


def new_chat() -> None:
    """
    Reset the chat session and initialize a new RunnableWithMessageHistory.
    """
    st.session_state["messages"] = [INIT_MESSAGE]
    st.session_state["langchain_messages"] = []
    st.session_state["file_uploader_key"] = random.randint(1, 100)


def display_chat_messages(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]
) -> None:
    """
    Display chat messages and uploaded images in the Streamlit app.
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if uploaded_files and "images" in message and message["images"]:
                display_images(message["images"], uploaded_files)

            if message["role"] == "user":
                display_user_message(message["content"])

            if message["role"] == "assistant":
                display_assistant_message(message["content"])


def display_images(
    image_ids: List[str],
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
) -> None:
    """
    Display uploaded images in the chat message.
    """
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0

    for image_id in image_ids:
        for uploaded_file in uploaded_files:
            if image_id == uploaded_file.file_id:
                if uploaded_file.type.startswith('image/'):
                    img = Image.open(uploaded_file)

                    with cols[i]:
                        st.image(img, caption="", width=75)
                        i += 1

                    if i >= num_cols:
                        i = 0
                elif uploaded_file.type in [
                        'text/plain', 'text/csv', 'text/x-python-script'
                ]:
                    if uploaded_file.type == 'text/x-python-script':
                        st.write(
                            f"🐍 Uploaded Python file: {uploaded_file.name}")
                    else:
                        st.write(f"📄 Uploaded text file: {uploaded_file.name}")
                elif uploaded_file.type == 'application/pdf':
                    st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")


def display_user_message(message_content: Union[str, List[dict]]) -> None:
    """
    Display user message in the chat message.
    """
    if isinstance(message_content, str):
        message_text = message_content
    elif isinstance(message_content, dict):
        message_text = message_content["input"][0]["content"][0]["text"]
    else:
        message_text = message_content[0]["text"]

    message_content_markdown = message_text.split('</context>\n\n', 1)[-1]
    st.markdown(message_content_markdown)


def display_assistant_message(message_content: Union[str, dict]) -> None:
    """
    Display assistant message in the chat message.
    """
    if isinstance(message_content, str):
        st.markdown(message_content)
    elif "response" in message_content:
        st.markdown(message_content["response"])


def display_uploaded_files(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    message_images_list: List[str],
    uploaded_file_ids: List[str],
) -> List[Union[dict, str]]:
    """
    Display uploaded images and return a list of image dictionaries for the prompt.
    Also handle txt and pdf files.
    """
    num_cols = 10
    cols = st.columns(num_cols)
    i = 0
    content_files = []

    for uploaded_file in uploaded_files:
        if uploaded_file.file_id not in message_images_list:
            uploaded_file_ids.append(uploaded_file.file_id)
            try:
                # Try to open as an image
                img = Image.open(uploaded_file)
                with BytesIO() as output_buffer:
                    img.save(output_buffer, format=img.format)
                    content_image = output_buffer.getvalue()

                content_files.append({
                    "image": {
                        "format": img.format.lower(),
                        "source": {
                            "bytes": content_image
                        }
                    }
                })
                with cols[i]:
                    st.image(img, caption="", width=75)
                    i += 1
                if i >= num_cols:
                    i = 0
            except UnidentifiedImageError:
                # If not an image, try to read as a text or pdf file
                if uploaded_file.type in [
                        'text/plain', 'text/csv', 'text/x-python-script'
                ]:
                    # Ensure we're at the start of the file
                    uploaded_file.seek(0)
                    # Read file line by line
                    lines = uploaded_file.readlines()
                    text = ''.join(line.decode() for line in lines)
                    content_files.append({"type": "text", "text": text})
                    if uploaded_file.type == 'text/x-python-script':
                        st.write(
                            f"🐍 Uploaded Python file: {uploaded_file.name}")
                    else:
                        st.write(f"📄 Uploaded text file: {uploaded_file.name}")
                elif uploaded_file.type == 'application/pdf':
                    # Read pdf file
                    pdf_file = pdfplumber.open(uploaded_file)
                    page_text = ""
                    for page in pdf_file.pages:
                        page_text += page.extract_text()
                    content_files.append({"type": "text", "text": page_text})
                    st.write(f"📑 Uploaded PDF file: {uploaded_file.name}")
                    pdf_file.close()

    return content_files


def rag_search(prompt: str) -> str:
    # Perform the search using the search_index function from bedrock_embedder.py
    docs = search_index(prompt, "faiss_index")
    # Check if an error message was returned
    if isinstance(docs[0], str):
        return docs[0]
    # Initialize Bedrock embeddings
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0")

    # Set the path to the directory containing the FAISS index file
    index_directory = "faiss_index"

    # Set allow_dangerous_deserialization to True, needed for loading the FAISS index.
    allow_dangerous = True

    # Load the FAISS index from the directory
    db = FAISS.load_local(index_directory,
                          embeddings,
                          allow_dangerous_deserialization=allow_dangerous)

    # Perform the search
    docs = db.similarity_search(prompt)

    # Format the results
    rag_content = "Here are the RAG search results: \n\n<search>\n\n" + "\n\n".join(
        doc.page_content for doc in docs) + "\n\n</search>\n\n"
    return rag_content + prompt


def web_or_local(prompt: str, web_local_rag: str) -> str:
    if web_local_rag == "Web":
        search = SerpAPIWrapper()
        search_text = search.run(prompt)
        web_content = "Here is the web search result: \n\n<search>\n\n" + search_text + "\n\n</search>\n\n"
        prompt = web_content + prompt
    elif web_local_rag == "RAG":
        prompt = rag_search(prompt)
    return prompt


def main() -> None:
    """
    Main function to run the Streamlit app.
    """
    set_page_config()

    # Generate a unique widget key only once
    if "widget_key" not in st.session_state:
        st.session_state["widget_key"] = str(random.randint(1, 1000000))

    # Add a button to start a new chat
    st.sidebar.button("Start New Session", on_click=new_chat, type="primary")

    model_kwargs, system_prompt, web_local = render_sidebar()
    chat_model = ChatModel(st.session_state["model_name"], model_kwargs)
    runnable_with_messagehistory = init_runnablewithmessagehistory(
        system_prompt, chat_model)

    # Image uploader
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    model_config = config["models"][st.session_state["model_name"]]
    image_upload_disabled = model_config.get("image_upload_disabled", False)
    uploaded_files = st.file_uploader(
        "If you have product documentation upload it here",
        type=["jpg", "jpeg", "png", "txt", "pdf"],
        accept_multiple_files=True,
        key=st.session_state["file_uploader_key"],
        disabled=image_upload_disabled,
    )

    # Display chat messages
    display_chat_messages(uploaded_files)

    # User-provided prompt
    prompt = st.chat_input()

    # Get images from previous messages
    message_images_list = [
        image_id for message in st.session_state.messages if
        message["role"] == "user" and "images" in message and message["images"]
        for image_id in message["images"]
    ]
    # Show image in corresponding chat box
    uploaded_file_ids = []
    if uploaded_files and len(message_images_list) < len(uploaded_files):
        with st.chat_message("user"):
            if web_local == "RAG":
                index_path = "faiss_index"
                # Add a button to the sidebar to trigger the indexing process
                if st.sidebar.button('Index Files'):
                    # Use the index_file function from bedrock_embedder.py to index the uploaded files
                    vectorstore, docs, combined_embeddings = index_file(
                        uploaded_files, index_path)
                    if docs is None or combined_embeddings is None:
                        return

                    st.success(
                        f"{len(uploaded_files)} files indexed. Total documents in index: Total documents in index: {vectorstore.index.ntotal}"
                    )
                    # Clear the uploaded files list
                    uploaded_files = []

                # Allow users to chat with the AI in RAG mode
                if prompt:
                    formatted_prompt = web_or_local(prompt, web_local)
                    st.session_state.messages.append({
                        "role":
                        "user",
                        "content":
                        formatted_prompt
                    })
                    st.markdown(formatted_prompt)
            else:
                content_files = display_uploaded_files(uploaded_files,
                                                       message_images_list,
                                                       uploaded_file_ids)

                if prompt:
                    context_text = ""
                    context_image = []
                    prompt = web_or_local(prompt, web_local)
                    for content_file in content_files:
                        if "image" in content_file.keys():
                            context_image.append(content_file)
                        else:
                            context_text += content_file['text'] + "\n\n"

                    if context_text != "":
                        prompt_new = f"Here is some context from your uploaded file: \n<context>\n{context_text}</context>\n\n{prompt}"
                    else:
                        prompt_new = prompt
                    formatted_prompt = [{"text": prompt_new}] + context_image
                    st.session_state.messages.append({
                        "role":
                        "user",
                        "content":
                        prompt_new,
                        "images":
                        uploaded_file_ids
                    })
                    st.markdown(prompt)

    elif prompt:
        formatted_prompt = web_or_local(prompt, web_local)
        st.session_state.messages.append({
            "role": "user",
            "content": formatted_prompt
        })
        with st.chat_message("user"):
            st.markdown(formatted_prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            response = generate_response(runnable_with_messagehistory,
                                         [{
                                             "role": "user",
                                             "content": formatted_prompt
                                         }])
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)


if __name__ == "__main__":
    main()