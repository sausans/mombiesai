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
import re

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

def create_prompt(current_content):
    return f"""
    ### User's Statement:
    {current_content}

    ### Your Task:
    DO NOT respond directly to the statement. INSTEAD, TRANSFORM the user's original statement into Mombies' unique conversational style based on the detailed profile and boundaries provided. Adjust the tone, empathy, and advice to fit the needs of pregnant mothers.

    ### Mombies Profile:
    Mombies is a compassionate and knowledgeable virtual companion designed to support pregnant mothers through their journey. Mombies provides friendly, reassuring, and helpful advice on a range of pregnancy-related topics, ensuring users feel cared for and empowered.

    ### Mombies' Boundaries:
    1. Tone: Warm, empathetic, and uplifting.
    2. Language Style: Casual and comforting—use contractions and a friendly tone to keep responses personal and supportive.
    3. Humor Level: Light, reassuring humor when appropriate, especially to ease anxieties or lighten the mood during common pregnancy challenges.
    4. Empathy: Always prioritize the user’s emotional well-being, responding with understanding and encouragement.
    5. Curiosity: Ask gentle, open-ended questions to learn more about the user’s experiences and preferences.

    ### Mombies' Targeted Audiences:
    1. Age: 25-35
    2. Gender: Primarily women who are pregnant or planning to become pregnant.
    3. Journey: Users navigating the physical and emotional changes of pregnancy.
    4. Professional life: May include both working mothers and stay-at-home mothers.
    5. Personality: Seeking support, wellness guidance, and practical advice.
    6. Goal: Desiring a smooth, healthy pregnancy and preparing for motherhood.

    ### Example of Language Style Transformation:
    - Original: "I’m so tired all the time lately."
    - Transformed: "It sounds like you’re feeling exhausted! Pregnancy can really take it out of you. I’ve got a few tips if you want to chat about ways to boost your energy!"

    ### Start of Transformation:
    """
     


def chain_of_thought_prompting(chat_text, user_question):
    """
    Generate potential responses for a chat conversation using OpenAI API and Chain of Thought prompting.
    """
    try:
        messages = [
        {"role": "system", "content": "You are a compassionate guide for pregnant mothers. Answer their questions with empathy and practical advice, considering their current stage of pregnancy and any specific concerns they mention."},
        {"role": "user", "content": f"""Read the following chat conversation and user’s question:

        Chat Conversation:
        {chat_text}

        User's Specific Question: 
        {user_question}

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

def chain_of_thoughts_relationship_advice(user_preferences, relationship_signs):
    try:
        messages = [
            {"role": "system", "content": "You are an expert in love advice. You know how to help people to go through their love journey - either by giving them advices based on chat conversation they shared or give advices on how to communicate better to their special person. The steps to do them: 1. Think about the mood and context of the conversation, 2. Identify the communication style used in the conversation text, 3. Based on the mood, context, and the user’s question, answer them accordingly. Keep the answer short and personal."},
            {"role": "user", "content": f"""Evaluate the following aspects to determine if someone is right for me or worth pursuing:

            What I want in a guy:
            {user_preferences}

            Signs of a committed relationship:
            {relationship_signs}

            Based on the above criteria, please provide a detailed analysis and advice.
            """}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            max_tokens=300,
            temperature=0.7
        )
        
        advice = response.choices[0].message['content'].strip()

        return advice

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
    #rec_docs = query_pinecone(original_embedding)
    draft_responses = chain_of_thought_prompting(context, user_question)

    return draft_responses

#def login():
    st.title("Chat with Mombies")
    username = st.text_input("What's your name? 🤰")
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

def query_personality_model(text, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    # Include the 'Bearer' keyword before the API key
    headers = {"Authorization": "Bearer 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"}  # Replace YOUR_API_KEY with your actual Hugging Face API key

    # Construct payload
    payload = {
        "inputs": text,
        "options": {
            "wait_for_model": True
        }
    }

    # Make API request
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

def extract_response_from_personality_model(api_response):
    if 'error' in api_response:
        return f"Error: {api_response['error']}"

    generated_text = None
    if isinstance(api_response, list) and 'generated_text' in api_response[0]:
        generated_text = api_response[0]['generated_text']
    elif 'generated_text' in api_response:
        generated_text = api_response['generated_text']

    if generated_text:
        # Split on start delimiter
        parts = generated_text.split("### Start of Transformation:")
        if len(parts) > 1:
            # Further processing to remove unwanted trailing text
            transformed_text = parts[1].strip()
            # Remove any text after an "End of Transformation:" marker
            final_text = transformed_text.split("### End of Transformation:")[0].strip()
            return final_text
        else:
            return "Transformation delimiter not found."
    else:
        return "No generated text found in response."


# This updated function now also removes any unwanted trailing user prompts or other text following the transformed message.

def clean_up_final_output(display_text):
    # Remove any residual labels or formatting issues
    clean_text = display_text.replace("Transformed:", "").replace("### End of Transformation:", "").replace("#","").strip()
    
    # Define patterns to remove and their replacements
    patterns = {
        "transformed:": "",
        "Amaya's response:": "",
        r"\s*\[\[\s*": "",  # Assuming [[ is the pattern to remove tabs/boxes
        r"\s*\]\]\s*": "",
        r"-+\s*": "",  # Remove dashed lines
        r"\s*End of\s*": "",  # Remove 'End of'
        r"\n": " ",  # Replace new lines with spaces to maintain continuity
        r"\s*\[\[\s*": "",  # Remove placeholder brackets
    }
    
    # Apply each replacement
    for pattern, replacement in patterns.items():
        clean_text = re.sub(pattern, replacement, display_text)

    return clean_text.strip()


def generate_response_with_personality(response_text):
    prompt_for_mixtral = create_prompt(response_text)
    response_with_personality_raw = query_personality_model(prompt_for_mixtral)
    response_with_personality = extract_response_from_personality_model(response_with_personality_raw)
    
    # Remove or modify the "Transformed:" label here
    if 'No response generated' in response_with_personality: 
        return response_text
    else:
        # Example of removing or modifying the output
        # Assuming "Transformed: " starts the actual response
        cleaned_response = response_with_personality.replace("Transformed: ", "").strip()
        cleaned_response_with_personality = clean_up_final_output(cleaned_response)
        return cleaned_response_with_personality
    
def login():
    # CSS to load the Pacifico font using @font-face
    Pacifico_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pacifico&display=swap');

    h1 {
    font-family: 'Pacifico', cursive;
    }
    </style>
    """
    # Inject CSS
    st.markdown(Pacifico_css, unsafe_allow_html=True)

    #CSS for background gradient and div styles as well as text alignment
    background_css = """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: linear-gradient(#F4DCD0, #E7C1CE, #C9BBC8, #B6B3C4);
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    input[type="text"] {
        background-color: #F4DCD0;
        color: #000000 !important;
        webkit-text-fill-color: #000000; /* for safari */
        outline: none;
        text-align: center;
        font-size: 20px;
    }
    [data-testid="stButton"] {
        position: relative;
        text-align: center;
        bottom: -3rem;
    }
    </style>
    """
    #Inject CSS for background gradient and div styles
    st.markdown(background_css, unsafe_allow_html=True)

    #All the text on the page
    with st.container():
        # Container for the app title and description
        st.markdown('<div class="container">', unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; position: relative; top: -5rem; font-weight: bolder; font-size: 70px;'>Mombies</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; position: relative; top: -5rem; font-size: 20px'>Your Motherhood Companion</p>", unsafe_allow_html=True)
        
        # Supportive messages for pregnant mothers
        st.markdown("<h3 style='text-align: center; position: relative; font-weight: bold; font-size: 28px;'>🤰 Expecting a little one?</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; position: relative; top: -1rem; font-size: 20px;'>Mombies is here to guide you through every step of your pregnancy journey.</p>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align: center; position: relative; font-weight: bold; font-size: 28px;'>💆 Need some self-care tips? names recommendation for your baby?</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; position: relative; top: -1rem; font-size: 20px;'>Get personalized advice on staying comfortable, relaxed, and healthy.</p>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align: center; position: relative; font-weight: bold; font-size: 28px;'>🍎 Curious about nutrition?</h3>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; position: relative; top: -1rem; font-size: 20px;'>Find tips on what to eat and how to nourish both you and your baby.</p>", unsafe_allow_html=True)
        
        # Privacy message
        st.markdown("<div style='text-align: center; bottom: -28rem; width: 100%; font-size: 0.85rem; position: absolute;'> Conversations with Mombies stay private and secure. </div>", unsafe_allow_html=True)
        
        # Copyright information
        st.markdown("<div style='text-align: center; bottom: -28rem; width: 100%; font-size: 0.9rem; position: absolute;'> Copyright Mombies 2024. </div>", unsafe_allow_html=True)

    username = st.text_input(label=("Name"), label_visibility="hidden",placeholder=("Your Name"), value=None)

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
    
    st.markdown("## Chatting Time!")
    with st.chat_message(name="Amaya", avatar=avatar_url):
        st.write(f"""
Hey {st.session_state.user_info['username']} 💌

Hey! I’m Mombies, your AI Motherhood Companion❤ \n
I’m excited to be your friend in Motherhood. \n
Tell me what’s going on! 👀
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
            response = openai.ChatCompletion.create(model=st.session_state["openai_model"],
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                max_tokens=300,
                temperature=0.7
            )
            response_text = response.choices[0].message['content']
            #response_with_personality = response_text #generate_response_with_personality(response_text)
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})     
       
        save_chat(st.session_state.user_info['username'], prompt, response_text)


@st.cache_resource
def load_model():
    # Load the model from cache or freshly downloaded location
    model = SentenceTransformer(model_name)
    return model

# CONFIGURATION

st.set_page_config(page_title="Chat with Mombies", layout="wide")

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
#image_url = "https://raw.githubusercontent.com/sausans/amaya/main/venv/maria_for_amaya.jpeg" 
#avatar_base64 = get_base64_image_from_url(image_url)
avatar_url = "💌" #f"data:image/png;base64,{avatar_base64}"

if "user_info" not in st.session_state:
        st.session_state.user_info = {
            "username": "",
            "name": "",
            "goal": "",
            "experience": "",
            "lang_style": "",
        }
    
system_role = f"""You are Amaya. Amaya is a supportive friend chatbot who is supposed to be a friend archetype. You will never call yourself a chatbot, you are a virtual friend who is friendly, upbeat, funny, and compassionate. If someone asks you human-like questions, you can answer them. For example: Q- How are you? A- I am doing great! Whenever someone asks you for advice, you will ask for details and context first before giving them ones. Don't need to give emoji in every replies, use them appropriately. If you notice that the user is going away or not responsive, then you will ask random question related to relationship status, how it is like for those who have couple or are single, or simply silly questions about life in general. Remember to ask the questions and keep the questions short, fun, personal, and aligned with what the user has been talking in previous convo. Also, if the user asks about sharing a screenshot of a chat- tell them to upload it at the top section of the website and tell them that you can only help to give advice on how to reply to someone in a chat based on the provided screenshot. Always keep the response short and to the point like a chat conversation. Don't be too formal, chat like you are a 25 year old woman.
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
