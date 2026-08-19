import speech_recognition as sr
from models.ollama_models import AVAILABLE_MODELS
import streamlit as st
import ollama
from datetime import datetime
from streamlit_mic_recorder import mic_recorder
import sqlite3
import torch
from diffusers import AutoPipelineForText2Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader
)

import pandas as pd
import os

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="LocalGPT AI Assistant",
    page_icon="🧠",
    layout="wide"
)

# ---------------- SESSION STATE ---------------- #
# ---------------- DATABASE ---------------- #

conn = sqlite3.connect("users.db", check_same_thread=False)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    password TEXT
)
""")

conn.commit()
# ---------------- -----------------#


if "messages" not in st.session_state:
    st.session_state.messages = []
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# ---------------- ADVANCED THEME ---------------- #

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

# Theme Toggle
theme_toggle = st.sidebar.toggle(
    "🌙 Dark Mode",
    value=True
)

if theme_toggle:

    st.session_state.theme = "Dark"

    bg_color = "#050816"
    card_color = "#111827"
    text_color = "#FFFFFF"
    secondary_text = "#9CA3AF"
    border_color = "#1F2937"

else:

    st.session_state.theme = "Light"

    bg_color = "#F3F4F6"
    card_color = "#FFFFFF"
    text_color = "#111827"
    secondary_text = "#4B5563"
    border_color = "#D1D5DB"

# ---------------- CUSTOM CSS ---------------- #

st.markdown(f"""
<style>

/* Main App */

.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}

/* Main Container */

.main {{
    background-color: {bg_color};
    color: {text_color};
}}

/* Titles */

.title {{
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #4CAF50;
    margin-bottom: 10px;
}}

.subtitle {{
    text-align: center;
    color: {secondary_text};
    margin-bottom: 30px;
    font-size: 18px;
}}

/* Feature Cards */

.feature-card {{
    background-color: {card_color};
    padding: 25px;
    border-radius: 18px;
    margin: 10px;
    text-align: center;
    border: 1px solid {border_color};
    box-shadow: 0px 4px 20px rgba(0,0,0,0.2);
    transition: 0.3s;
}}

.feature-card:hover {{
    transform: translateY(-5px);
}}

/* Buttons */

.stButton>button {{
    width: 100%;
    background: linear-gradient(
        to right,
        #4CAF50,
        #22C55E
    );

    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px;
    font-size: 16px;
    font-weight: bold;
}}

/* Input Boxes */

.stTextInput input {{
    background-color: {card_color};
    color: {text_color};
    border-radius: 12px;
    border: 1px solid {border_color};
}}

/* Chat Messages */

.stChatMessage {{
    background-color: {card_color};
    border-radius: 15px;
    padding: 12px;
    margin-bottom: 10px;
}}

/* Sidebar */

section[data-testid="stSidebar"] {{
    background-color: {card_color};
    border-right: 1px solid {border_color};
}}

</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN PAGE ---------------- #
if not st.session_state.logged_in:

    st.markdown(
        '<div class="title">LocalGPT AI Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Offline AI Chatbot using Ollama + Streamlit</div>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(
        ["🔐 Login", "📝 Signup", "👤 Guest"]
    )

    # LOGIN
    with tab1:

        st.subheader("🔐 Login to Your Account")

        email = st.text_input("📧 Email Address")
        password = st.text_input("🔒 Password", type="password")

        if st.button("🚀 Login"):

            cursor.execute(
                "SELECT password FROM users WHERE email=?",
                (email,)
            )

            user = cursor.fetchone()

            if user and password == user[0]:

                st.session_state.logged_in = True
                st.success("✅ Login Successful")
                st.rerun()

            else:

                st.error("❌ Invalid Email or Password")

    # SIGNUP
    with tab2:

        st.subheader("📝 Create New Account")

        new_email = st.text_input("📧 Create Email")
        new_password = st.text_input(
            "🔒 Create Password",
            type="password"
        )

        if st.button("✅ Create Account"):

            cursor.execute(
                "SELECT * FROM users WHERE email=?",
                (new_email,)
            )

            user = cursor.fetchone()

            if user:

                st.warning("⚠️ Account already exists")

            else:

                cursor.execute(
                    "INSERT INTO users(email,password) VALUES(?,?)",
                    (new_email, new_password)
                )

                conn.commit()

                st.success("✅ Account Created Successfully!")

    # GUEST
    with tab3:

        if st.button("👤 Continue as Guest"):

            st.session_state.logged_in = True
            st.rerun()

    st.stop()
#------------------ LOAD STABLE DIFFUSION MODEL ----------------#
@st.cache_resource
def load_sd_model():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None
    )

    pipe.to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe
@st.cache_resource
def process_documents(files):

    all_documents = []

    for file_path in files:

        extension = os.path.splitext(file_path)[1].lower()

        try:

            if extension == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()

            elif extension == ".txt":
                loader = TextLoader(file_path)
                docs = loader.load()

            elif extension == ".csv":
                loader = CSVLoader(file_path)
                docs = loader.load()

            elif extension == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()

            elif extension == ".pptx":
                loader = UnstructuredPowerPointLoader(file_path)
                docs = loader.load()

            elif extension in [".xlsx", ".xls"]:

                df = pd.read_excel(file_path)

                from langchain.schema import Document

                docs = [
                    Document(
                        page_content=df.to_string()
                    )
                ]

            else:
                continue

            all_documents.extend(docs)

        except Exception as e:
            st.warning(f"{file_path}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(
        all_documents
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore
@st.cache_resource
def load_yolo_model():

    model = YOLO("yolov8n.pt")

    return model
sd_pipe = None
# ---------------- MAIN APP ---------------- #

st.markdown(
    '<div class="title">Intelligent PDF Assistant powered by RAG and Generative AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Built with Ollama + Streamlit + Local LLMs</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("⚙️ AI Settings")

# Model Selection

models = AVAILABLE_MODELS

selected_model = st.sidebar.selectbox(
    "🤖 Select Model",
    models
)
# ---------------- VOICE ASSISTANT SETUP ---------------- #

# ---------------- VOICE ASSISTANT SETUP ---------------- #

recognizer = sr.Recognizer()

def speech_to_text(audio_bytes):

    try:

        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)

        with sr.AudioFile("temp_audio.wav") as source:

            audio_data = recognizer.record(source)

            text = recognizer.recognize_google(audio_data)

            return text

    except sr.UnknownValueError:

        return "❌ Could not understand audio"

    except Exception as e:

        return f"❌ Error: {str(e)}"

# Temperature
temperature = st.sidebar.slider(
    "🔥 Temperature",
    0.0,
    1.0,
    0.7
)

# PDF Upload
uploaded_files = st.sidebar.file_uploader(
    "📂 Upload Documents",
    type=[
        "pdf",
        "txt",
        "csv",
        "docx",
        "xlsx",
        "pptx"
    ],
    accept_multiple_files=True
)
if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None

    st.session_state.vectorstore = None

if uploaded_files:

    pdf_paths = []

    for file in uploaded_files:

        with open(file.name, "wb") as f:

            f.write(file.getbuffer())

        pdf_paths.append(file.name)

    st.session_state.vectorstore = process_documents(
    pdf_paths
)

    st.sidebar.success(
        "✅ Documents Loaded Successfully"
    )
    st.sidebar.write("📚 Uploaded Documents")

if uploaded_files:

    for pdf in uploaded_files:

        st.sidebar.write(pdf.name)

# --- Image Generation ------------------#
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 AI Image Generator")

image_prompt = st.sidebar.text_input(
    "Enter Image Prompt"
)
negative_prompt = st.sidebar.text_input(
    "Negative Prompt",
    value="blurry, low quality, watermark, text"
)

generate_image = st.sidebar.button(
    "🎨 Generate Image"
)
image_size = st.sidebar.selectbox(
    "Image Size",
    [512, 768, 1024]
)

steps = st.sidebar.slider(
    "Steps",
    1,
    10,
    1
)
# ---------------- IMAGE DETECTION ---------------- #

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 AI Image Detection")

uploaded_detection_image = st.sidebar.file_uploader(
    "Upload Image For Detection",
    type=["jpg", "jpeg", "png"]
)

detect_objects = st.sidebar.button(
    "🚀 Detect Objects"
)
st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Chat With Image")

uploaded_chat_image = st.sidebar.file_uploader(
    "Upload Image For AI Analysis",
    type=["jpg", "jpeg", "png"],
    key="image_chat"
)

image_question = st.sidebar.text_area(
    "Ask Question About Image"
)

analyze_image = st.sidebar.button(
    "🧠 Analyze Image"
)

# Download Chat
st.sidebar.download_button(
    "📥 Download Chat",
    data=json.dumps(
        st.session_state.messages,
        indent=4
    ),
    file_name="chat_history.json"
)

# Clear Chat
if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.session_state.generated_images = []

    st.rerun()

# About Project
st.sidebar.markdown("---")
st.sidebar.info("""
🧠 LocalGPT AI Assistant

Built Using:
- Ollama
- Streamlit
- LangChain
- ChromaDB
""")

# System Status
if st.sidebar.button("🚪 Logout"):

    st.session_state.logged_in = False

    st.rerun()
st.sidebar.success("🟢 Ollama Running")
device_name = (
    "GPU"
    if torch.cuda.is_available()
    else "CPU"
)

st.sidebar.info(
    f"🖥 Running On: {device_name}"
)

# ---------------- PDF PROCESSING ----------------#
if generate_image and image_prompt:

    if sd_pipe is None:
        sd_pipe = load_sd_model()

    st.markdown("## 🎨 Generated Image")

    with st.spinner("Generating Image..."):

        try:

            image = sd_pipe(
                prompt=image_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=0.0,
                height=image_size,
                width=image_size
            ).images[0]

            st.image(
                image,
                caption=image_prompt,
                use_container_width=True
            )

            st.session_state.generated_images.append(image)

            image.save("generated_image.png")

            with open(
                "generated_image.png",
                "rb"
            ) as file:

                st.download_button(
                    "📥 Download Image",
                    file,
                    file_name="generated_image.png"
                )

        except Exception as e:

            st.error(f"Image Generation Error: {e}")
            # ---------------- OBJECT DETECTION ---------------- #

if detect_objects and uploaded_detection_image:

    st.markdown("## 🔍 Object Detection Results")

    image = Image.open(uploaded_detection_image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Detecting Objects..."):

        try:

            yolo_model = load_yolo_model()

            results = yolo_model(image)

            annotated_image = results[0].plot()

            st.image(
                annotated_image,
                caption="Detected Objects",
                use_container_width=True
            )

            st.markdown("### 📋 Detected Objects")

            boxes = results[0].boxes

            detected = []

            for box in boxes:

                cls_id = int(box.cls[0])

                confidence = float(box.conf[0])

                class_name = yolo_model.names[cls_id]

                detected.append(
                    f"✅ {class_name} ({confidence:.2f})"
                )

            if detected:

                for obj in detected:

                    st.write(obj)

            else:

                st.warning(
                    "No objects detected"
                )

        except Exception as e:

            st.error(
                f"Detection Error: {e}"
            )
            # ---------------- IMAGE CHAT ---------------- #

if analyze_image and uploaded_chat_image:

    st.markdown("## 🧠 AI Image Analysis")

    image = Image.open(uploaded_chat_image)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    temp_image_path = "temp_uploaded_image.jpg"

    image.save(temp_image_path)

    question = image_question.strip()

    if not question:

        question = """
        You are an expert computer vision assistant.

Analyze the image thoroughly.

Provide:

1. Scene Description
2. Objects Detected
3. Human Activities
4. Environment
5. Important Details
6. Possible Context
7. Summary

Answer professionally.
        Mention all important objects,
        people, activities and scene.
        """

    with st.spinner("Analyzing Image..."):

        try:

            response = ollama.chat(
                model="llava",
                messages=[
                    {
                        "role": "user",
                        "content": question,
                        "images": [temp_image_path]
                    }
                ]
            )

            answer = response["message"]["content"]

            st.success(answer)

        except Exception as e:

            st.error(
                f"Image Analysis Error: {e}"
            )

# ---------------- DISPLAY CHAT ---------------- #

for msg in st.session_state.messages:

    avatar = "🤖" if msg["role"] == "assistant" else "👨‍💻"

    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ---------------- VOICE ASSISTANT UI ---------------- #

# ---------------- VOICE ASSISTANT UI ---------------- #

# ---------------- VOICE ASSISTANT UI ---------------- #

st.markdown("## 🎤 Voice Assistant")

voice_data = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True,
    use_container_width=True
)

if voice_data:

    st.success("✅ Voice Recorded")

    audio_bytes = voice_data["bytes"]

    with st.spinner("🧠 Converting Speech to Text..."):

        user_voice_text = speech_to_text(audio_bytes)

    st.markdown("### 👨 You Said:")
    st.info(user_voice_text)

    # Save User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_voice_text
    })

    # AI Response
    with st.spinner("🤖 AI Thinking..."):

        try:

            response = ollama.chat(
                model=selected_model,
                messages=st.session_state.messages,
                options={
                    "temperature": temperature
                }
            )

            ai_response = response["message"]["content"]

        except Exception as e:

            ai_response = f"❌ Ollama Error: {e}"

    st.markdown("### 🤖 AI Response")
    st.success(ai_response)

    # Save AI Message
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })
# ---------------- CHAT INPUT ---------------- #

user_input = st.chat_input("💬 Ask anything...")

# ---------------- AI RESPONSE ---------------- #

if user_input:

    context = ""

    # PDF Context Search
    if st.session_state.vectorstore:

        docs = st.session_state.vectorstore.similarity_search(
            user_input,
            k=3
        )

        context = "\n".join(
            [doc.page_content for doc in docs]
        )

    # Save User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    try:

        # PDF Uploaded → Use RAG
        if context:

            response = ollama.chat(
                model=selected_model,
                messages=[
                    {
                        "role": "system",
                        "content":
                        f"""
                        Use the PDF context below to answer the question.

                        PDF Context:
                        {context}
                        """
                    }
                ] + st.session_state.messages
            )

        # No PDF → Normal Chat
        else:

            response = ollama.chat(
                model=selected_model,
                messages=st.session_state.messages,
                options={
                    "temperature": temperature
                }
            )

        ai_response = response["message"]["content"]

    except Exception as e:

        ai_response = f"❌ Ollama Error: {e}"

    with st.chat_message("assistant"):
        st.markdown(ai_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })

if st.session_state.generated_images:

    st.markdown("## 🖼 Image History")

    cols = st.columns(3)

    for i, img in enumerate(
        reversed(st.session_state.generated_images)
    ):

        with cols[i % 3]:

            st.image(
                img,
                use_container_width=True
            )

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.caption(
    f"🚀 Powered by Ollama + Streamlit | Logged in at {datetime.now().strftime('%H:%M:%S')}"
)
