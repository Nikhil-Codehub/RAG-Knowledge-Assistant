---

# 🧠 RAG Knowledge Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge\&logo=python\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge\&logo=langchain\&logoColor=white)
![Groq](https://img.shields.io/badge/Groq%20AI-Fastest-orange?style=for-the-badge)

**RAG Knowledge Assistant** ek advanced document-chat AI tool hai jisme aap apni **PDF, DOCX, TXT** files ko upload karke unse baat kar sakte ho.
Ye **Retrieval-Augmented Generation (RAG)** use karta hai jisse answers sirf document ke content par based hote hain — hallucinations zero.
Groq ke ultra-fast **LPU hardware** par chalne ki wajah se responses ChatGPT se bhi fast milte hain.

---

## ✨ Features

* 📚 **Chat With Your Files:** PDF, DOCX, TXT — sab ek saath upload karo
* ⚡ **ChatGPT-style Streaming Output:** Typing effect ke saath live response
* 🔍 **Accurate Citations:** Har answer ke end me → *Source: file.pdf • Page 3*
* 🧠 **Conversational Memory:** Follow-up questions me automatically context use karta hai
* 🤖 **Multi-Model Selection:**

  * Llama-3.1 8B
  * Mixtral-8x7B
  * Gemma-2 9B
* ⚙ **Local Vector DB (FAISS):** Super fast document search
* 🐳 **Docker Ready:** One-click deployment

---

## 🛠 Tech Stack

| Component  | Technology       |
| ---------- | ---------------- |
| Frontend   | Streamlit        |
| LLM        | Groq API         |
| RAG Engine | LangChain        |
| Vector DB  | FAISS            |
| Embeddings | all-MiniLM-L6-v2 |
| Deployment | Docker           |

---

## ⚙ Architecture (Simple Flow)

```
User Query → Retriever → Groq LLM → Streaming Output
      ↑            ↓
  Chat History   FAISS Vector Store
                      ↑
              Embeddings + Chunking
                      ↑
                Document Upload
```

---

## 🚀 Setup (Local)

### 1. Clone Repo

```bash
git clone https://github.com/Nikhil-Codehub/RAG-Knowledge-Assistant.git
cd RAG-Knowledge-Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv myenv
# Windows
myenv\Scripts\activate
# Mac/Linux
source myenv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Add API Key

Create `.env`:

```
GROQ_API_KEY=gsk_your_key_here
```

### 5. Run App

```bash
streamlit run app.py
```

---

## 🐳 Run with Docker

### Build

```bash
docker build -t rag-assistant .
```

### Run

```bash
docker run -p 8501:8501 rag-assistant
```

---

## 📁 Project Structure

```
📁 RAG-Knowledge-Assistant
 ├── app.py
 ├── .env.example
 ├── Dockerfile
 ├── requirements.txt
 ├── README.md
 ├── /data
 └── /vectorstore
```

---

## 💡 Future Add-ons

* Export chat as PDF
* Multiple document workspaces
* Redis Vector Store
* UI Theme Toggle

---
