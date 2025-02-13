import streamlit as st
import arxiv
import requests
import fitz
from io import BytesIO
import fpdf
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="PaperPal",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'selected_paper' not in st.session_state:
    st.session_state.selected_paper = None
if 'papers' not in st.session_state:
    st.session_state.papers = []

def create_chat_pdf():
    """Generate PDF of chat history"""
    pdf = fpdf.FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Research Paper Analysis Chat History", ln=True, align='C')
    
    # Paper details
    if st.session_state.selected_paper:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Paper: {st.session_state.selected_paper.title}", ln=True)
        pdf.cell(0, 10, f"Authors: {', '.join([a.name for a in st.session_state.selected_paper.authors])}", ln=True)
        pdf.cell(0, 10, f"Date Generated: {datetime.now().strftime('%B %d, %Y %H:%M:%S')}", ln=True)
    
    pdf.ln(10)
    
    # Chat messages
    for message in st.session_state.chat_history:
        # Message header
        pdf.set_font("Arial", "B", 11)
        role = "User" if message["role"] == "user" else "Assistant"
        pdf.cell(0, 10, f"{role}:", ln=True)
        
        # Message content
        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 10, message["content"])
        pdf.ln(5)
    
    return pdf.output(dest='S').encode('latin1')

def search_papers(query, search_type="keyword", max_results=5):
    """Search for papers on arXiv"""
    try:
        if search_type == "title":
            search = arxiv.Search(
                query=f'ti:"{query}"',
                max_results=1
            )
        else:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
        
        return list(search.results())
    except Exception as e:
        st.error(f"Error searching papers: {str(e)}")
        return []

def extract_text_from_pdf(pdf_url):
    """Extract text from PDF URL"""
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        
        return text
    except Exception as e:
        st.error(f"Error extracting PDF text: {str(e)}")
        return None

def process_paper(paper):
    """Process selected paper for analysis"""
    try:
        with st.spinner("Extracting text from paper..."):
            text = extract_text_from_pdf(paper.pdf_url)
            if not text:
                return False
        
        with st.spinner("Processing paper content..."):
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            
            # Create embeddings and vector store
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            docs = [Document(page_content=chunk) for chunk in chunks]
            st.session_state.vector_db = FAISS.from_documents(docs, embeddings)
            
            return True
    except Exception as e:
        st.error(f"Error processing paper: {str(e)}")
        return False

def initialize_qa_chain():
    """Initialize the QA chain"""
    template = """You are an expert research paper analyst. Use the following pieces of context to provide a detailed answer to the question. If you can't answer the question based on the context, say so.

Context:
{context}

Question: {question}

Previous conversation:
{chat_history}

Please provide a comprehensive response that:
1. Directly answers the question using information from the paper
2. Cites specific sections or findings when relevant
3. Explains technical concepts clearly
4. Uses examples when helpful
5. Maintains academic accuracy while being accessible

Answer:"""

    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite-preview-02-05",
        temperature=0.7
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vector_db.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def display_paper_card(paper, index=None):
    """Display paper details in a card format"""
    with st.container():
        st.markdown(f"### {paper.title}")
        st.markdown(f"**Authors:** {', '.join([a.name for a in paper.authors])}")
        st.markdown(f"**Published:** {paper.published.strftime('%B %d, %Y')}")
        st.markdown(f"**Categories:** {', '.join(paper.categories)}")
        
        with st.expander("View Abstract"):
            st.markdown(paper.summary)
        
        if index is not None:
            if st.button("Select Paper", key=f"select_{index}"):
                st.session_state.selected_paper = paper
                with st.spinner("Processing paper..."):
                    if process_paper(paper):
                        st.session_state.chat_history = []
                        st.rerun()

# Main UI
st.title("PaperPal🐧")

# Sidebar for search
with st.sidebar:
    st.header("Paper Search")
    search_type = st.radio("Search by:", ["Keywords", "Title"])
    search_query = st.text_input("Enter search terms:")
    
    if st.button("Search", use_container_width=True):
        if search_query:
            with st.spinner("Searching papers..."):
                st.session_state.papers = search_papers(
                    search_query,
                    "title" if search_type == "Title" else "keyword"
                )

# Main content area
if not st.session_state.selected_paper:
    # Display search results
    if st.session_state.papers:
        st.header("Search Results")
        for i, paper in enumerate(st.session_state.papers):
            display_paper_card(paper, i)
            st.divider()
    else:
        st.info(" Use the sidebar to search for a research paper")

else:
    # Display selected paper and chat interface
    st.header("Selected Paper")
    display_paper_card(st.session_state.selected_paper)
    
    # Download button for chat history
    if st.session_state.chat_history:
        chat_pdf = create_chat_pdf()
        st.download_button(
            "📥 Download Chat History (PDF)",
            data=chat_pdf,
            file_name=f"research_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
        )

    st.header("💬 Ask Questions")
    
    # Initialize QA chain
    qa_chain = initialize_qa_chain()

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the paper..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing paper..."):
                response = qa_chain({"question": prompt})
                st.markdown(response["answer"])
                
                # Show sources
                if response.get("source_documents"):
                    with st.expander("View Sources"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(doc.page_content)
                            st.divider()
        
        # Add assistant response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response["answer"]
        })

    # Reset button
    if st.button("Start New Analysis"):
        st.session_state.selected_paper = None
        st.session_state.chat_history = []
        st.session_state.vector_db = None
        st.rerun()