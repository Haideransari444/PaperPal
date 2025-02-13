import arxiv
import requests
import fitz  # PyMuPDF
import os
from typing import List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

@dataclass
class ResearchPaper:
    """Data class to store research paper metadata and content."""
    title: str
    authors: List[str]
    published: str
    pdf_url: str
    content: Optional[str] = None

class ResearchAssistant:
    """A class to handle research paper search, processing, and interaction."""
    
    def __init__(self):
        """Initialize the research assistant with necessary components."""
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        self.embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.vector_db = None
        
    def search_paper(self) -> Optional[ResearchPaper]:
        """Interactive search for research papers with improved error handling."""
        while True:
            try:
                print("\nSearch Options:")
                print("1. Search by Exact Title")
                print("2. Search by Keywords")
                print("3. Exit Search")
                
                choice = input("\nEnter your choice (1-3): ").strip()
                
                if choice == "3":
                    return None
                    
                if choice not in ["1", "2"]:
                    print("Invalid choice. Please enter 1, 2, or 3.")
                    continue
                
                # Handle exact title search
                if choice == "1":
                    title = input("Enter the exact title of the paper: ").strip()
                    search = arxiv.Search(
                        query=f'ti:"{title}"',
                        max_results=1
                    )
                
                # Handle keyword search
                else:
                    keywords = input("Enter keywords to search for papers: ").strip()
                    search = arxiv.Search(
                        query=keywords,
                        max_results=5
                    )
                
                papers = list(search.results())
                
                if not papers:
                    print("\nNo papers found matching your criteria.")
                    continue
                
                # For keyword search, let user select from results
                if choice == "2":
                    print("\nFound Papers:")
                    for i, paper in enumerate(papers, 1):
                        print(f"{i}. {paper.title}")
                        print(f"   Authors: {', '.join(a.name for a in paper.authors)}")
                        print(f"   Published: {paper.published.strftime('%Y-%m-%d')}")
                        print()
                    
                    selection = int(input("\nSelect a paper (1-5) or 0 to search again: ").strip())
                    if selection == 0:
                        continue
                    if not (1 <= selection <= len(papers)):
                        print("Invalid selection.")
                        continue
                    paper = papers[selection - 1]
                else:
                    paper = papers[0]
                
                return ResearchPaper(
                    title=paper.title,
                    authors=[a.name for a in paper.authors],
                    published=paper.published.strftime("%Y-%m-%d"),
                    pdf_url=paper.pdf_url
                )
                
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("Please try again.")
    
    def process_paper(self, paper: ResearchPaper) -> bool:
        """Process the paper: download PDF, extract text, create embeddings."""
        try:
            # Download and extract text
            paper.content = self._extract_text_from_pdf(paper.pdf_url)
            
            # Split text into chunks
            chunks = self._split_text(paper.content)
            
            # Generate embeddings and create vector database
            self._create_vector_db(chunks)
            
            return True
            
        except Exception as e:
            print(f"Error processing paper: {str(e)}")
            return False
    
    def _extract_text_from_pdf(self, pdf_url: str) -> str:
        """Extract text from PDF without saving to disk."""
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text("text"))
        
        return "\n".join(text_parts)
    
    def _split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return splitter.split_text(text)
    
    def _create_vector_db(self, chunks: List[str]):
        """Create FAISS vector database from text chunks."""
        docs = [Document(page_content=chunk) for chunk in chunks]
        self.vector_db = FAISS.from_documents(docs, self.embeddings_model)
    
    def query_paper(self, query: str) -> str:
        """Query the paper using the conversation chain."""
        if not self.vector_db:
            raise ValueError("No paper has been processed yet.")
            
        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vector_db.as_retriever(),
            memory=self.memory
        )
        
        response = qa_chain({"question": query})
        return response["answer"]

def main():
    """Main function to demonstrate the research assistant's capabilities."""
    assistant = ResearchAssistant()
    
    print("Welcome to the Research Paper Assistant!")
    
    # Search for paper
    paper = assistant.search_paper()
    if not paper:
        print("Search cancelled.")
        return
        
    print(f"\nSelected paper: {paper.title}")
    print("Processing paper...")
    
    # Process the paper
    if not assistant.process_paper(paper):
        print("Failed to process paper.")
        return
        
    print("\nPaper processed successfully!")
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question about the paper (or 'quit' to exit): ").strip()
        if query.lower() == 'quit':
            break
            
        try:
            answer = assistant.query_paper(query)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()