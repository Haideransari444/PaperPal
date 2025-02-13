import streamlit as st
import arxiv
import requests
import fitz  # PyMuPDF
import os
from io import BytesIO
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from fpdf import FPDF

# Load API keys
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

def search_research_paper(title):
    """Fetches the first matching research paper from ArXiv."""
    search = arxiv.Search(query=f'ti:"{title}"', max_results=1)
    papers = list(search.results())
    return papers[0] if papers else None

def extract_text_from_pdf(pdf_url):
    """Streams a PDF from a URL and extracts text."""
    response = requests.get(pdf_url, stream=True)
    if response.status_code != 200:
        return None
    doc = fitz.open(stream=BytesIO(response.content), filetype="pdf")
    return "\n".join([page.get_text("text") for page in doc])

def split_paper_text(text):
    """Splits text into manageable chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text(text)

def generate_embeddings(chunks):
    """Generates embeddings for text chunks and stores them in FAISS."""
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = [Document(page_content=chunk) for chunk in chunks]
    return FAISS.from_documents(docs, embeddings_model)

# Streamlit UI
st.title("AI Research Paper Chat Assistant")

# Paper Search
paper_title = st.text_input("Enter the Research Paper Title:")
if st.button("Fetch Paper"):
    paper = search_research_paper(paper_title)
    if paper:
        st.success(f"Found: {paper.title} by {', '.join([a.name for a in paper.authors])}")
        st.write(f"[Download PDF]({paper.pdf_url})")
        pdf_text = extract_text_from_pdf(paper.pdf_url)
        if pdf_text:
            st.session_state["paper_text"] = pdf_text
            st.success("Text extracted successfully!")
        else:
            st.error("Failed to extract text from PDF.")
    else:
        st.error("No matching paper found.")

# AI Chat Interface
if "paper_text" in st.session_state:
    chunks = split_paper_text(st.session_state["paper_text"])
    vector_db = generate_embeddings(chunks)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", temperature=0.5)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, vector_db.as_retriever(), memory=memory)
    
    st.subheader("Chat with the Paper")
    user_query = st.text_input("Ask a question:")
    if st.button("Submit Query"):
        response = qa_chain.run(user_query)
        st.session_state.setdefault("chat_history", []).append((user_query, response))
        st.write(response)
    
    # Export Chat to PDF
    if st.button("Export Chat to PDF"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "AI Research Chat History", ln=True, align="C")
        pdf.ln(10)
        for q, a in st.session_state["chat_history"]:
            pdf.multi_cell(0, 10, f"Q: {q}\nA: {a}\n", border=0)
            pdf.ln()
        pdf_output = BytesIO()
        pdf.output(pdf_output, "F")
        st.download_button("Download Chat PDF", pdf_output.getvalue(), "chat_history.pdf", "application/pdf")
