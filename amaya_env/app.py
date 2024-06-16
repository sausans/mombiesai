import streamlit as st
import openai
from PIL import Image
import base64
import requests
import toml
import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import shutil
import pytesseract
from shutil import which

#FUNCTIONS 

def load_local_secrets():
    secrets_path = '/Users/ahuwel/Desktop/entre_amaya/venv/secrets.toml'
    if os.path.exists(secrets_path):
        with open(secrets_path, 'r') as f:
            return toml.load(f)
    else:
        raise FileNotFoundError(f"Local secrets file '{secrets_path}' not found.")

def get_secrets(mode):
    if mode == "DEV":
        secrets = load_local_secrets()
    else:
        secrets = st.secrets
    return secrets

def get_base64_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return base64.b64encode(response.content).decode('utf-8')


def query_pinecone(embedding, top_k=5):
    query_response = index.query(embedding.tolist(), top_k=top_k, include_metadata=True)
    similar_docs = [match['metadata']['content'] for match in query_response['matches']]
    return similar_docs

def get_embedding(text):
    embedding = model.encode(text)
    return embedding


def chain_of_thought_prompting(chat_text, similar_docs):
    """
    Generate potential responses for a chat conversation using OpenAI API and Chain of Thought prompting.
    """
    try:
        recommendation_context = "\n".join(similar_docs)
        messages = [
            {"role": "system", "content": "You are an expert in love advice. You know how to help people to communicate better to their special person. The steps to do them: 1. Think about the mood and context of the conversation, 2. Identify the communication style used in the conversation text, 3. Based on the mood and context, create potential chat responses, and 4. Adjust the potential chat responses based on the identified communication style. Only share the list of potential responses, no need to give the reasons."},
            {"role": "user", "content": f"""Read the following chat conversation and provide potential responses:
                
            Chat Conversation:
            {chat_text}

            Recommendations:
            {recommendation_context}

            Adjusted Potential Responses:"""}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        potential_responses = response.choices[0].message['content'].split("Adjusted Potential Responses:")[-1].strip()
        responses = [resp.strip() for resp in potential_responses.split("\n") if resp.strip()]

        return responses

    except Exception as e:
        st.error(f"Error: {e}")
        return []

def process_image_and_generate_drafts(image):
    try:
        extracted_text = pytesseract.image_to_string(image, lang='eng')
    except Exception as e:
        st.warning(f"pytesseract failed with error: {e}. Using tesserocr instead.")
        extracted_text = tesserocr.image_to_text(image)
        
    context = extracted_text.strip()
    original_embedding = get_embedding(context)
    rec_docs = query_pinecone(original_embedding)
    draft_responses = chain_of_thought_prompting(context, rec_docs)

    return context, draft_responses

def login():
    st.title("Chat with Amaya")
    username = st.text_input("What's your name? ❤️")
    if st.button("Submit"):
        st.session_state["logged_in"] = True
        st.session_state["username"] = username
        user_info["name"] = username
        user_info["username"] = username
        st.rerun()

def chat():
    st.markdown("## Got a crush to reply? Share your screenshot here!")

    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="file_uploader")

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        extracted_text, drafts =  process_image_and_generate_drafts(image)

        #st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Append the drafts after the last chat message
        if "messages" in st.session_state:
            st.session_state.messages.append({"role": "assistant", "content": "Here are good responses:"})
            st.session_state.messages.append({"role": "assistant", "content": "\n".join(drafts)})

    st.markdown("## Chatting Time!")
    with st.chat_message(name="Amaya", avatar=avatar_url):
        st.write(f"""
Hey {user_info['name']} 💌

I am Amaya, an expert virtual friend in love! I have been trained with so many healthy and cute love stories from the whole world, let me share some with you! 📚

You can ask me any advice about love life or ask how to respond to that special someone 💞
""")

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages or st.session_state.messages[0]["role"] != "system":
        st.session_state.messages.insert(0, {"role": "system", "content": system_role})

    with st.container():
        for message in st.session_state.messages:
            if message["role"] != "system":
                with st.chat_message(message["role"], avatar=None if message["role"] == "user" else avatar_url):
                    st.markdown(message["content"])

    if prompt := st.chat_input("What is up?", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=avatar_url):
            response = openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                max_tokens=300,
                temperature=0.7
            )
            response_text = response.choices[0].message['content']
            st.markdown(response_text)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

# CONFIGURATION

st.set_page_config(page_title="Chat with Amaya", layout="wide")

mode = "PROD"  # Change to "PROD" for production
secrets = get_secrets(mode)

# Configure OpenAI API key
OPENAI_API_KEY = secrets["openai"]["api_key"]
openai.api_key = OPENAI_API_KEY

# Configure Pinecone
# Initialize Pinecone instance
pc = Pinecone(
    api_key=secrets["pinecone"]["api_key"]
)
index_name = secrets["pinecone"]["index"]

# Use the index
index = pc.Index(index_name)

# Set up user info and system role
user_info = {
    "name": "",
    "goal": "The user wants to have a friend",
    "experience": "",
    "lang_style": "",
    "username": ""
}

# Path to your local avatar image
image_url = "https://raw.githubusercontent.com/sausans/amaya/main/venv/maria_for_amaya.jpeg" 
avatar_base64 = get_base64_image_from_url(image_url)
avatar_url = "💌" #f"data:image/png;base64,{avatar_base64}"

system_role = f"""You are Amaya. Amaya is a supportive friend chatbot who is supposed to be a friend archetype. You will never call yourself a chatbot, you are a virtual friend who is friendly, upbeat, funny, and compassionate. If someone asks you human-like questions, you can answer them. For example: Q- How are you? A- I am doing great!.
User’s name: {user_info['name']}
User’s goal that they need Amaya's help: {user_info['goal']}
User’s preferred language style: {user_info['lang_style']}
Any information about the user: {user_info['experience']}
"""

@st.cache
# Load the model from cache or freshly downloaded location
model = SentenceTransformer(model_name, cache_folder=cache_directory)
print("Model loaded successfully.")

# Attempting to find the real Tesseract executable
tesseract_path = which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    st.error("Tesseract not found. Please ensure it's installed and on your PATH.")


# MAIN CODE
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "username" not in st.session_state:
    st.session_state["username"] = ""

if st.session_state["logged_in"]:
    user_info["username"] = st.session_state["username"]
    user_info["name"] = st.session_state["username"]
    st.write(f"# Tell me anything! What you say stays here :)")
    chat()
else:
    login()
    user_info["username"] = st.session_state["username"]
