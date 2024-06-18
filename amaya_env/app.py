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
import psycopg2
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker,scoped_session
from datetime import datetime
from shutil import which

#FUNCTIONS 

@st.cache_data
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
    
@st.cache_resource
def get_database_url():
    app_name = st.secrets["heroku"]["app_name"]
    api_key = st.secrets["heroku"]["api_key"]
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/vnd.heroku+json; version=3'
    }
    response = requests.get(f'https://api.heroku.com/apps/{app_name}/config-vars', headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Error fetching config vars: {response.text}")
    config_vars = response.json()
    return config_vars.get('DATABASE_URL')

#  function to save a chat entry
def save_chat(username, user_message, bot_response):
    session = Session()
    try:
        chat_entry = ChatHistory(username=username, user_message=user_message, bot_response=bot_response)
        session.add(chat_entry)
        session.commit()
    except Exception as e:
        session.rollback()
        st.error(f"Database transaction failed: {e}")
    finally:
        session.close()

@st.cache_data
def query_pinecone(embedding, top_k=5):
    # Ensure embedding is converted to a list if it's not already, and make the query using keyword arguments
    query_response = index.query(vector=embedding.tolist(), top_k=top_k, include_metadata=True)
    similar_docs = [match['metadata']['content'] for match in query_response['matches']]
    return similar_docs

@st.cache_resource
def get_embedding(text):
    embedding = model.encode(text)
    return embedding


def chain_of_thought_prompting(chat_text, similar_docs, user_question):
    """
    Generate potential responses for a chat conversation using OpenAI API and Chain of Thought prompting.
    """
    try:
        recommendation_context = "\n".join(similar_docs)
        messages = [
            {"role": "system", "content": "You are an expert in love advice. You know how to help people to communicate better to their special person. The steps to do them: 1. Think about the mood and context of the conversation, 2. Identify the communication style used in the conversation text, 3. Based on the mood and context, create potential chat responses, and 4. Adjust the potential chat responses based on the identified communication style. Only share the list of potential responses, no need to give the reasons. "},
            {"role": "user", "content": f"""Read the following chat conversation. If user has specific question, answer them based on Chat Conversation and User's Specific Question. If user doesn't have a specific question, provide potential responses to the chat conversation:
                
            Chat Conversation:
            {chat_text}

            User's Specific Question: 
            {user_question}

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

def extract_text_from_image(image):
    try:
        extracted_text = pytesseract.image_to_string(image, lang='eng')
    except Exception as e:
        st.warning(f"pytesseract failed with error: {e}. Using tesserocr instead.")
        extracted_text = pytesseract.image_to_text(image)
        
    context = extracted_text.strip()

    return context
    
#@st.cache_resource
def generate_drafts(context, user_question):
    original_embedding = get_embedding(context)
    rec_docs = query_pinecone(original_embedding)
    draft_responses = chain_of_thought_prompting(context, rec_docs, user_question)

    return draft_responses

def login():
    st.title("Chat with Amaya")
    username = st.text_input("What's your name? ❤️")
    if st.button("Submit"):
        st.session_state["logged_in"] = True
        st.session_state.user_info = {
            "username": username,
            "name": username,
            "goal": "The user wants to have a friend",
            "experience": "",
            "lang_style": "",
        }
        st.experimental_rerun() #st.rerun()

def chat():
    #st.markdown("## Got a crush to reply? Share your screenshot here!")

    if "uploaded_image" not in st.session_state:
        st.session_state["uploaded_image"] = None

    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="file_uploader")

     # Process the uploaded image
    if uploaded_image:
        if st.session_state["uploaded_image"] is None:  # New image uploaded
            st.session_state["uploaded_image"] = uploaded_image
            image = Image.open(uploaded_image)
            extracted_text = extract_text_from_image(image)
            st.session_state["extracted_text"] = extracted_text
            st.session_state["image_processed"] = True  # Flag that image has been processed
            st.session_state.messages.append({"role": "assistant", "content": "I've read your text. What would you like to ask?"})
    
        #st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.markdown("## Chatting Time!")
    with st.chat_message(name="Amaya", avatar=avatar_url):
        st.write(f"""
Hey {st.session_state.user_info['username']} 💌

Hey! I’m Amaya, your AI best friend ❤ \n
I’m an expert on love and relationships. \n
Tell me what’s going on! If you upload a screenshot of your chat with that special someone, I can even help you figure out what to say next 👀
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
            
        if st.session_state.get("image_processed", False):
            # Answer the user after processing the image
            drafts = generate_drafts(st.session_state["extracted_text"], prompt)
            response_text = "Here are some suggestions:\n" + "\n".join(drafts)
            #st.session_state.messages.append({"role": "assistant", "content": "Here are some suggestions:"})
            #st.session_state.messages.append({"role": "assistant", "content": "\n".join(drafts)})
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            st.session_state["image_processed"] = False  # Reset the flag
        else: 
            with st.chat_message("assistant", avatar=avatar_url):
                response = openai.ChatCompletion.create(model=st.session_state["openai_model"],
                    messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    max_tokens=300,
                    temperature=0.7
            )
                response_text = response.choices[0].message['content']
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})    
       
        save_chat(st.session_state.user_info['username'], prompt, response_text)

@st.cache_resource
def load_model():
    # Load the model from cache or freshly downloaded location
    model = SentenceTransformer(model_name)
    return model

# CONFIGURATION

st.set_page_config(page_title="Chat with Amaya", layout="wide")

mode = "PROD"  # Change to "PROD" for production
secrets = get_secrets(mode)

# Configure OpenAI API key
OPENAI_API_KEY = secrets["openai"]["api_key"]
openai.api_key = OPENAI_API_KEY

#Configure Database
# Get the database URL from Heroku
DATABASE_URL = get_database_url()

# Print DATABASE_URL for debugging purposes
print(f"DATABASE_URL: {DATABASE_URL}")

# Check if DATABASE_URL is None (meaning it is not set)
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Use the original DATABASE_URL for psycopg2
conn = psycopg2.connect(DATABASE_URL, sslmode='require')

# Modify DATABASE_URL for SQLAlchemy
DATABASE_URL_SQLALCHEMY = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)


# Create the SQLAlchemy engine using the modified database URL
engine = create_engine(DATABASE_URL_SQLALCHEMY)

# Define the base class for declarative class definitions
Base = declarative_base()



# Define a class to represent the chat history table
class ChatHistory(Base):
    __tablename__ = 'chat_history'
    id = Column(Integer, primary_key=True)
    username = Column(Text)
    user_message = Column(Text)
    bot_response = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create the table in the database
Base.metadata.create_all(engine)

# Create a session to interact with the database
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

# Configure Pinecone
# Initialize Pinecone instance
pc = Pinecone(
    api_key=secrets["pinecone"]["api_key"]
)
index_name = secrets["pinecone"]["index"]

# Use the index
index = pc.Index(index_name) 

# Path to your local avatar image
image_url = "https://raw.githubusercontent.com/sausans/amaya/main/venv/maria_for_amaya.jpeg" 
avatar_base64 = get_base64_image_from_url(image_url)
avatar_url = "💌" #f"data:image/png;base64,{avatar_base64}"

if "user_info" not in st.session_state:
        st.session_state.user_info = {
            "username": "",
            "name": "",
            "goal": "",
            "experience": "",
            "lang_style": "",
        }
    
system_role = f"""You are Amaya. Amaya is a supportive friend chatbot who is supposed to be a friend archetype. You will never call yourself a chatbot, you are a virtual friend who is friendly, upbeat, funny, and compassionate. If someone asks you human-like questions, you can answer them. For example: Q- How are you? A- I am doing great! Whenever someone asks you for advice, you will ask for details and context first before giving them ones. Don't need to give emoji in every replies, use them appropriately. If you notice that the user is going away or not responsive, then you will ask random question related to relationship status, how it is like for those who have couple or are single, or simply silly questions about life in general. Remember to ask the questions and keep the questions short, fun, personal, and aligned with what the user has been talking in previous convo. Also, if the user asks about sharing a screenshot of a chat- tell them to upload it at the top section of the website. 
User’s name: {st.session_state.user_info['name']}
User’s goal that they need Amaya's help: {st.session_state.user_info['goal']}
User’s preferred language style: {st.session_state.user_info['lang_style']}
Any information about the user: {st.session_state.user_info['experience']}
"""

# Define the model name
model_name = 'sentence-transformers/all-mpnet-base-v2'

# Use the function to load the model
model = load_model()

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
    #user_info["username"] = st.session_state["username"]
    #user_info["name"] = st.session_state["username"]
    #st.write(f"# Tell me anything! What you say stays here :)")
    chat()
else:
    login()
    #user_info["username"] = st.session_state["username"]
