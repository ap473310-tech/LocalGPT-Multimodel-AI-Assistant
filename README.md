# LocalGPT AI Assistant

## Project Overview

LocalGPT AI Assistant is an enterprise-grade offline-first Generative AI platform developed using Streamlit, Ollama, LangChain, FAISS, Stable Diffusion, YOLOv8, and Local Large Language Models.

The application enables users to interact with AI through multiple modalities including text, voice, documents, and images while running completely on local infrastructure without relying on cloud-based AI services.

The project combines Conversational AI, Retrieval-Augmented Generation (RAG), Computer Vision, Image Generation, Speech Processing, and Multi-Document Intelligence into a single unified platform.

---

## Problem Statement

Most AI assistants depend heavily on cloud APIs, which introduces several challenges:

* Privacy concerns
* Internet dependency
* High API costs
* Limited customization
* Data security risks

This project addresses these issues by providing a fully local AI ecosystem powered by Ollama and open-source models.

---

# Key Features

## 1. Local LLM Chat Assistant

Users can chat with multiple locally running LLMs through Ollama.

### Features

* Offline AI conversations
* Multiple model selection
* Temperature control
* Chat history management
* Real-time responses

### Technologies

* Ollama
* Streamlit

---

## 2. Multi-Document RAG System

Users can upload multiple documents and ask questions directly from their content.

### Supported Formats

* PDF
* TXT
* CSV
* DOCX
* XLSX
* PPTX

### Workflow

1. Document Upload
2. Text Extraction
3. Text Chunking
4. Embedding Generation
5. Vector Storage (FAISS)
6. Similarity Search
7. Context Retrieval
8. LLM Response Generation

### Technologies

* LangChain
* FAISS
* Sentence Transformers
* Recursive Character Text Splitter

---

## 3. Voice Assistant

Users can interact with the assistant using voice commands.

### Capabilities

* Speech-to-Text
* AI Response Generation
* Text-to-Speech

### Workflow

Voice Input → Speech Recognition → LLM Processing → Voice Response

### Technologies

* SpeechRecognition
* Pyttsx3
* Streamlit Mic Recorder

---

## 4. AI Image Generation

Generate images using natural language prompts.

### Features

* Text-to-Image Generation
* Adjustable Image Size
* Negative Prompt Support
* Image Download

### Technologies

* Stable Diffusion Turbo
* Diffusers
* PyTorch

---

## 5. Object Detection System

Detect and identify objects from uploaded images using YOLOv8.

### Features

* Real-time Object Detection
* Bounding Boxes
* Confidence Scores
* Annotated Image Output

### Technologies

* YOLOv8
* OpenCV
* PIL

---

## 6. Vision AI (Chat With Image)

Users can upload images and ask questions about them.

### Example Queries

* What objects are present?
* Describe the scene.
* What activity is happening?
* Summarize the image.

### Workflow

Image Upload → LLaVA Vision Model → Image Understanding → AI Response

### Technologies

* LLaVA
* Ollama

---

## 7. User Authentication System

Provides secure access using login and signup functionality.

### Features

* User Registration
* Login
* Guest Mode
* Session Management

### Technologies

* SQLite

---

## 8. Chat Management

### Features

* Chat History
* Download Chat History
* Clear Chat
* Session Storage

---

## 9. Dark/Light Theme Support

### Available Modes

* Dark Mode
* Light Mode

Provides better accessibility and user experience.

---

# System Architecture

```text
                    User
                      │
      ┌───────────────┼───────────────┐
      │               │               │
      ▼               ▼               ▼

  Text Chat      Voice Input      Image Input
      │               │               │
      ▼               ▼               ▼

   Ollama      Speech Recognition   LLaVA
      │
      ▼

   AI Response

------------------------------------------------

Document Upload
      │
      ▼

Document Loader
      │
      ▼

Text Splitter
      │
      ▼

Embeddings
      │
      ▼

FAISS Vector Store
      │
      ▼

Similarity Search
      │
      ▼

Context Retrieval
      │
      ▼

Local LLM
      │
      ▼

Final Answer
```

---

# Tech Stack

## Frontend

* Streamlit

## Backend

* Python

## LLM Layer

* Ollama
* Llama Models
* Mistral Models
* Gemma Models
* LLaVA

## RAG

* LangChain
* FAISS
* Sentence Transformers

## Computer Vision

* YOLOv8
* OpenCV

## Image Generation

* Stable Diffusion Turbo
* Diffusers

## Voice AI

* SpeechRecognition
* Pyttsx3

## Database

* SQLite

---

# RAG Pipeline

```text
Document Upload
      │
      ▼

Document Loading
      │
      ▼

Text Chunking
      │
      ▼

Embedding Generation
      │
      ▼

FAISS Vector Store
      │
      ▼

Similarity Search
      │
      ▼

Context Retrieval
      │
      ▼

Ollama LLM
      │
      ▼

Final Answer
```

---

# Folder Structure

```text
LocalGPT-AI-Assistant/

├── app.py
├── users.db
├── requirements.txt
├── README.md
├── chat_history.json
│
├── generated_images/
├── uploaded_documents/
│
├── models/
│   └── ollama_models.py
│
└── assets/
    ├── screenshots/
    └── icons/
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-repository/localgpt-ai-assistant.git

cd localgpt-ai-assistant
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Start Ollama

```bash
ollama serve
```

## Download Models

```bash
ollama pull llama3

ollama pull mistral

ollama pull llava
```

## Run Application

```bash
streamlit run app.py
```

---

# Future Enhancements

* Agentic AI Integration
* Multi-Agent Workflow
* SQL Database Chat
* Video Understanding
* OCR Support
* Real-Time Webcam Detection
* Hybrid Search
* Milvus Integration
* Cloud Deployment
* User Role Management

---

# Interview Explanation

LocalGPT AI Assistant is a multimodal Generative AI platform that integrates local LLMs, Retrieval-Augmented Generation (RAG), Computer Vision, Speech Processing, and Image Generation into a single application.

The frontend is built using Streamlit, while Ollama is used for running local Large Language Models. For document intelligence, a complete RAG pipeline is implemented using LangChain, Sentence Transformers, and FAISS. Users can upload PDFs, DOCX, PPTX, CSV, Excel, and TXT files, and the system retrieves relevant document chunks before generating responses.

The platform also includes multimodal capabilities such as Stable Diffusion Turbo for image generation, YOLOv8 for object detection, and LLaVA for image understanding. Voice interaction is enabled through SpeechRecognition and Pyttsx3, while SQLite handles user authentication and session management.

The primary objective of the project is to create a privacy-focused, offline-first AI assistant capable of handling text, documents, voice, and images within a unified local AI ecosystem.
