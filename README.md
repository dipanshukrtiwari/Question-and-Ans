# RAG-Based Question Answering System
 (docmindragai)
 

## 1. Project Overview
The RAG-Based Question Answering System allows users to upload documents and ask questions based on their content. The system retrieves relevant information from the uploaded document and generates context-based answers.

## 2. Objectives
- To extract information from uploaded documents.
- To answer questions using document context.
- To reduce incorrect or unsupported answers.
- To provide a simple user interface for document-based question answering.

## 3. Key Features
- Document upload
- Document text processing
- Question answering using retrieved context
- Responses based on uploaded documents
- Handling of questions whose answers are not available in the document

## 4. Technologies Used
- Python
- Streamlit
- Retrieval-Augmented Generation (RAG)
- Embeddings and a language model, depending on the project configuration

## 5. How to Run
1. Open the project folder in the terminal.
2. Activate the Python virtual environment.
3. Install the dependencies using `pip install -r requirements.txt`.
4. Run the application using `streamlit run app.py`.
5. Open the local URL displayed in the terminal.

## 6. Testing Results
- Document upload: Successful
- Expected question-answer tests: Passed
- Answers grounded in uploaded documents: Verified
- Missing information response: Verified

## 7. Conclusion
The project demonstrates how Retrieval-Augmented Generation can be used to answer questions from uploaded documents. The application was tested for document upload, answer correctness, and handling of unavailable information.
