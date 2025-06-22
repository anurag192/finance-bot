import os
import pickle
from flask import Flask, render_template, request, redirect, url_for, flash
from dotenv import load_dotenv

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Load environment variables (e.g. GOOGLE_API_KEY)
load_dotenv()

app = Flask(__name__)


# Paths & constants
VECTORSTORE_PATH = "faiss_store.pkl"

# Initialise embeddings & LLM once (Gemini 1.5 Flash)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    urls = [request.form.get(f"url{i}") for i in range(1, 4)]
    urls = [u for u in urls if u]

    if not urls:
        
        return redirect(url_for("index"))

    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        separators=["\n\n", "\n", ".", ","]
    )
    chunks = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    with open(VECTORSTORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    flash(f" Successfully indexed {len(chunks)} text chunks from {len(urls)} URL(s).", "success")
    return redirect(url_for("index"))

@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query", "").strip()

    if not query:
        flash(" Please enter a question.", "danger")
        return redirect(url_for("index"))

    if not os.path.exists(VECTORSTORE_PATH):
        flash(" Vector store not found. Please process URLs first.", "danger")
        return redirect(url_for("index"))

    with open(VECTORSTORE_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    result = chain({"question": query}, return_only_outputs=True)

    answer = result.get("answer", "No answer generated.")
    sources = [s for s in result.get("sources", "").split("\n") if s]

    return render_template("index.html", answer=answer, sources=sources, query=query)

if __name__ == "__main__":
    app.run(debug=True)
