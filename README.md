# 🪨  News Research Tool

This is a user-friendly and intelligent news research assistant designed to simplify the process of extracting financial and stock market insights from online articles.  
It combines the power of LangChain, GoogleGenerativeAI embeddings, and FAISS to deliver fast and accurate answers to user queries based on real-time article content.

---

## 🔍 Features

- 📥 Load article URLs directly or upload a `.txt` file containing multiple URLs.
- 📰 Extract and parse article content using **LangChain's UnstructuredURLLoader**.
- 🧠 Generate vector embeddings with **OpenAI's Embeddings API**.
- ⚡ Perform efficient similarity search using **FAISS (Facebook AI Similarity Search)**.
- 💬 Ask contextual questions and get AI-powered answers with references to **source URLs**.
- 💾 FAISS index is saved in a local `.pkl` file for future reuse and fast lookups.

---

## 🛠️ Installation

### 1. Clone the repository:
```bash
https://github.com/anurag192/finance-bot.git
cd finance project
```

### 2. Install dependencies:
```bash
# Install all the required Python packages
pip install -r requirements.txt
```

### 3. Set up your OpenAI API key:
Create a `.env` file in the root directory with the following content:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## 🚀 Usage

### 1. Launch the app:
```bash
streamlit run main.py
```

### 2. Workflow:

- Use the sidebar to:
  - Paste one or more URLs directly, **or**
  - Upload a `.txt` file containing a list of URLs.

- Click on **"Process URLs"** to:
  - Load and extract article content.
  - Split content into manageable chunks.
  - Generate OpenAI embeddings.
  - Store and index them using FAISS.

- Once processing is complete:
  - Enter your query in the text box.
  - Receive an AI-generated answer with **source links**.

---

## 📁 Project Structure

```bash
finance project/
├── main.py                  # Streamlit app for UI and logic
├── requirements.txt         # List of Python packages
├── faiss_index            # Auto-generated FAISS index file
├── .env                     # OpenAI API key (user-supplied)
```



## 💡 Notes

- Ensure your OpenAI API key is valid and has API access to embedding models.
- FAISS index will be reused across sessions to speed up retrieval.
- Restarting the app will still retain the index if `faiss_store_openai.pkl` exists.


