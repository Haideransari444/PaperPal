# PaperPal: AI-Powered Research Paper Assistant

PaperPal is a **Streamlit-based** AI-powered research assistant that helps users search for, analyze, and interact with research papers from **arXiv**. It extracts paper content, generates embeddings using **LangChain**, and allows users to ask contextual questions with a memory-enabled **Conversational Retrieval Chain**.



![image](https://github.com/user-attachments/assets/0b41b8af-8823-40bc-9f37-1a400be6abdc)
![image](https://github.com/user-attachments/assets/0235a727-acaa-4d59-ac9b-69b3524e9f70)
![image](https://github.com/user-attachments/assets/5e80d61a-7212-4787-b925-185691872205)


## 🚀 Features

- **🔍 Search Research Papers:** Find papers on arXiv using keywords or titles.
- **📄 Extract PDF Content:** Automatically extract and process research paper text.
- **🧠 AI-Powered Q&A:** Ask contextual questions about the paper.
- **📚 Memory-Powered Chat:** Retains conversation history for better responses.
- **📥 Download Chat History:** Save and download your chat in a PDF format.

## 🛠️ Technologies Used

- **Python**
- **Streamlit** (Frontend UI)
- **arXiv API** (Paper search)
- **PyMuPDF (fitz)** (PDF extraction)
- **LangChain** (Conversational AI)
- **Google Generative AI** (Embeddings & LLM)
- **FAISS** (Vector database for search)
- **FPDF** (PDF generation)

## 📌 Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/paperpal.git
   cd paperpal
   ```

2. Create and activate a virtual environment:

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```sh
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file:
     ```sh
     touch .env
     ```
   - Add your API keys (Google AI, etc.) inside `.env`:
     ```env
     GOOGLE_API_KEY=your_google_api_key
     ```

## 🚀 Running the Application

Start the Streamlit app:

```sh
streamlit run app.py
```

## 📖 How It Works

1. **Search for a Paper**: Enter a keyword or title to find relevant research papers.
2. **Select & Process the Paper**: Click on a paper to extract text and generate AI embeddings.
3. **Chat with the Paper**: Ask research questions, and AI will provide detailed answers.
4. **Download Chat History**: Save your conversation in a **PDF format** for future reference.

## 🛠️ Future Enhancements

-

## 🤝 Contributing

Feel free to submit issues, feature requests, or contribute via pull requests!

## 📜 License

This project is licensed under the **MIT License**.

---

**Made with ❤️ **
