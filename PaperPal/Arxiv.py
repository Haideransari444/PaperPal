import arxiv
import requests
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


def search_research_paper():
    """
    Allows the user to search for research papers either by exact title or keywords.
    If the user selects keywords, the function will display the top 5 matching papers
    and allow the user to choose one.
    """

    print("How would you like to search for the research paper?")
    print("1. Search by Exact Title")
    print("2. Search by Keywords")
    
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        title = input("Enter the exact title of the paper: ").strip()
        search = arxiv.Search(
            query=f'ti:"{title}"',  # Search by exact title
            max_results=1
        )
        papers = list(search.results())

        if papers:
            paper = papers[0]
            print("\nPaper Found:")
            print(f"Title: {paper.title}")
            print(f"Authors: {', '.join([a.name for a in paper.authors])}")
            print(f"Published: {paper.published}")
            print(f"PDF Link: {paper.pdf_url}")
            return paper.pdf_url
        else:
            print("\nNo paper found with that exact title.")

    elif choice == "2":
        keyword = input("Enter keywords to search for papers: ").strip()
        search = arxiv.Search(
            query=keyword,  # Search by keywords
            max_results=5
        )
        papers = list(search.results())

        if papers:
            print("\nTop 5 matching papers:")
            for i, paper in enumerate(papers):
                print(f"{i+1}. {paper.title} - {', '.join([a.name for a in paper.authors])}")

            selection = int(input("\nEnter the number of the paper you want to choose (1-5): ").strip()) - 1
            if 0 <= selection < len(papers):
                selected_paper = papers[selection]
                print("\nYou selected:")
                print(f"Title: {selected_paper.title}")
                print(f"Authors: {', '.join([a.name for a in selected_paper.authors])}")
                print(f"Published: {selected_paper.published}")
                print(f"PDF Link: {selected_paper.pdf_url}")
                return selected_paper.pdf_url
            else:
                print("\nInvalid selection.")
        else:
            print("\nNo papers found for the given keywords.")

    else:
        print("\nInvalid choice. Please enter 1 or 2.")

# Example Usage:
pdf_url = search_research_paper()



def extract_text_from_pdf_stream(pdf_url):
    """Streams a PDF from a URL and extracts text without saving it."""
    
    # Fetch the PDF as a stream
    response = requests.get(pdf_url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch PDF. Status code: {response.status_code}")
    
    # Load PDF into memory
    pdf_stream = BytesIO(response.content)
    
    # Open PDF with PyMuPDF
    doc = fitz.open(stream=pdf_stream, filetype="pdf")
    
    # Extract text from all pages
    extracted_text = "\n".join([page.get_text("text") for page in doc])
    
    return extracted_text

text = extract_text_from_pdf_stream(pdf_url)


def split_paper_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits the research paper text into manageable chunks.
    
    :param text: The full text of the research paper
    :param chunk_size: Maximum number of characters per chunk
    :param chunk_overlap: Number of overlapping characters between chunks
    :return: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)




def split_paper_text(text, chunk_size=1000, chunk_overlap=100):
    """
    Splits the research paper text into manageable chunks.
    
    :param text: The full text of the research paper
    :param chunk_size: Maximum number of characters per chunk
    :param chunk_overlap: Number of overlapping characters between chunks
    :return: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)

# Example usage:
paper_text = text
chunks = split_paper_text(paper_text)

print(f"Total chunks: {len(chunks)}")
print("First chunk:", chunks[10])


def generate_and_store_embeddings(chunks):
    """
    Generates embeddings for each text chunk and stores them in a FAISS vector database.
    """
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  
   
    # Convert chunks to LangChain document format
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Create a FAISS vector database
    vector_db = FAISS.from_documents(docs, embeddings_model)
    print("DONE")

    return vector_db

# Example usage:
vector_db = generate_and_store_embeddings(chunks)

def query_vector_db(query):
    """Search the vector database for relevant chunks."""
    results = vector_db.similarity_search(query, k=3)  # Retrieve top 3 results
    return results

# Example usage
user_query = "Applications of LLM in software"
retrieved_chunks = query_vector_db(user_query)

# Print the results
for i, chunk in enumerate(retrieved_chunks):
    print(f"Result {i+1}:\n{chunk.page_content}\n")



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite-preview-02-05", temperature=0.7)  # Use "gemini-pro-vision" for multimodal input


qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=vector_db.as_retriever(), memory=memory
)

def chat_with_paper(query):
    response = qa_chain.invoke({"question": query})
    return response["answer"]

while True:
    query = input("\nAsk a question about the paper (or type 'exit' to quit): ").strip()
    if query.lower() == "exit":
        print("Exiting chat...")
        break
    response = chat_with_paper(query)
    print("\nAI Response:", response)