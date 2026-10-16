
import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="DocuMind AI | Smart PDF Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# PREMIUM UI STYLES
# ==================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #5B6CFF;
    --primary-dark: #4052E8;
    --ink: #172033;
    --muted: #64748B;
    --border: #E2E8F0;
    --surface: #FFFFFF;
}

.stApp {
    background: #F4F7FC;
    color: var(--ink);
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] .main .block-container {
    max-width: 1180px;
    padding: 2rem 2rem 4rem 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0B1220;
    border-right: 1px solid #202C41;
}

[data-testid="stSidebar"] > div:first-child {
    background: #0B1220;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #E5ECF7 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #9BAAC0 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #27344A;
}

/* Main text defaults */
.stApp p,
.stApp li,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] {
    color: #172033;
}

.stApp h1,
.stApp h2,
.stApp h3 {
    color: #172033;
    letter-spacing: -0.5px;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(125deg, #101A32 0%, #283B80 55%, #625BEB 100%);
    border-radius: 24px;
    padding: 36px;
    margin-bottom: 28px;
    box-shadow: 0 16px 42px rgba(48, 61, 150, 0.17);
}

.hero-kicker {
    color: #C7D2FE !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 15px;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(28px, 4vw, 43px);
    line-height: 1.18;
    font-weight: 800;
    margin: 0 0 15px 0;
    letter-spacing: -1.3px;
}

.hero p {
    color: #E0E7FF !important;
    font-size: 16px;
    line-height: 1.8;
    max-width: 640px;
    margin: 0;
}

.hero-pill {
    display: inline-block;
    margin-top: 22px;
    padding: 8px 13px;
    background: rgba(255,255,255,0.12);
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 30px;
    font-size: 12px;
    font-weight: 600;
}

/* Cards */
.surface-card {
    background: #FFFFFF;
    border: 1px solid #E3E9F3;
    border-radius: 18px;
    padding: 22px;
    box-shadow: 0 5px 20px rgba(22, 34, 60, 0.035);
    margin-bottom: 15px;
}

.surface-card h3 {
    color: #172D55 !important;
    margin-top: 0;
}

.surface-card p {
    color: #64748B !important;
    line-height: 1.75;
}

.card-icon {
    font-size: 25px;
    margin-bottom: 12px;
}

.section-heading {
    font-size: 22px;
    font-weight: 800;
    color: #172033 !important;
    margin-bottom: 4px;
}

.section-subtitle {
    color: #718096 !important;
    font-size: 14px;
    margin-bottom: 18px;
}

/* Upload */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 2px dashed #A5B4FC;
    border-radius: 18px;
    padding: 18px;
    transition: border-color .2s ease, background .2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #5B6CFF;
    background: #FAFAFF;
}

[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span {
    color: #334155 !important;
}

/* Buttons */
.stButton > button {
    background: #5B6CFF !important;
    color: #FFFFFF !important;
    border: 1px solid #5B6CFF !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    min-height: 42px;
    transition: all .2s ease;
}

.stButton > button:hover {
    background: #4052E8 !important;
    border-color: #4052E8 !important;
    color: #FFFFFF !important;
    box-shadow: 0 5px 14px rgba(91, 108, 255, 0.2);
}

/* Chat message cards */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important;
    border: 1px solid #E0E7F1 !important;
    border-radius: 17px !important;
    padding: 17px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 4px 18px rgba(22, 34, 60, 0.035);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    color: #172033 !important;
    line-height: 1.85 !important;
}

[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color: #3147BD !important;
}

/* Answer title */
.answer-label {
    color: #4052E8 !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #FFFFFF !important;
    border: 1px solid #D8E1F0 !important;
    border-radius: 16px !important;
    box-shadow: 0 5px 22px rgba(22, 34, 60, 0.055);
}

[data-testid="stChatInput"] textarea {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    caret-color: #5B6CFF !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8794A9 !important;
    -webkit-text-fill-color: #8794A9 !important;
}

/* Text inputs */
.stTextInput input {
    color: #172033 !important;
    -webkit-text-fill-color: #172033 !important;
    background: #FFFFFF !important;
    border: 1px solid #D8E1F0 !important;
    border-radius: 10px !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 13px !important;
}

/* Status badge */
.status-note {
    background: #ECFDF5;
    border: 1px solid #A7F3D0;
    color: #047857 !important;
    border-radius: 12px;
    padding: 12px 15px;
    font-size: 13px;
    font-weight: 600;
    margin: 12px 0 20px 0;
}

/* Small text */
.muted {
    color: #718096 !important;
    font-size: 14px;
    line-height: 1.7;
}

#MainMenu, footer {
    visibility: hidden;
}

@media (max-width: 700px) {
    [data-testid="stAppViewContainer"] .main .block-container {
        padding: 1rem 1rem 3rem 1rem;
    }

    .hero {
        padding: 24px;
        border-radius: 18px;
    }

    .hero h1 {
        font-size: 29px;
    }
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE
# ==================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

if "indexed_bytes" not in st.session_state:
    st.session_state.indexed_bytes = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("## ✨ DocuMind AI")
    st.caption("INTELLIGENT DOCUMENT WORKSPACE")
    st.divider()

    st.markdown("### 🔐 API Configuration")

    gemini_api_key = (
        st.secrets.get("GEMINI_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
    )

    if not gemini_api_key:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Your key is used to access the Gemini API."
        )

    if gemini_api_key:
        st.success("API key available")
    else:
        st.warning("API key required")

    st.divider()
    st.markdown("### 📄 Knowledge Base")

    if st.session_state.indexed_file:
        st.success("Document indexed")
        st.caption(st.session_state.indexed_file)
    else:
        st.info("No PDF uploaded yet")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset document", use_container_width=True):
        st.session_state.indexed_file = None
        st.session_state.indexed_bytes = None
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### ⚙️ Technology")
    st.caption("Retrieval-Augmented Generation")
    st.caption("FAISS Vector Search")
    st.caption("HuggingFace Embeddings")
    st.caption("Google Gemini")
    st.divider()
    st.caption("Made for smarter document exploration.")


# ==================================================
# HERO
# ==================================================
st.markdown("""
<div class="hero">
    <div class="hero-kicker">✦ Your personal AI researcher</div>
    <h1>Your documents.<br>Your answers.</h1>
    <p>
        Turn your PDF into an interactive knowledge base.
        Ask questions in natural language and explore the information
        that matters to you.
    </p>
    <div class="hero-pill">📚 PDF Intelligence &nbsp; • &nbsp; 🤖 AI-powered answers</div>
</div>
""", unsafe_allow_html=True)


# ==================================================
# EMBEDDINGS
# ==================================================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )


# ==================================================
# CREATE VECTOR STORE
# ==================================================
@st.cache_resource(show_spinner=False)
def create_vectorstore(file_bytes):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs or not any(
            doc.page_content.strip() for doc in docs
        ):
            raise ValueError("No readable text found in this PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No usable text chunks were extracted.")

        embeddings = load_embeddings()
        db = FAISS.from_documents(chunks, embeddings)

        return db, len(docs), len(chunks)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==================================================
# UPLOAD SECTION
# ==================================================
st.markdown(
    '<div class="section-heading">📂 Your knowledge base</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">Start by uploading a PDF document.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose your PDF document",
    type=["pdf"],
    help="Upload a readable, text-based PDF."
)

if uploaded_file is None:
    st.markdown("""
    <div class="surface-card">
        <div class="card-icon">🚀</div>
        <h3>Ready to explore your documents?</h3>
        <p>
            Upload your PDF above to discover key information,
            understand long documents, and ask specific questions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="surface-card">
            <div class="card-icon">🔎</div>
            <h3>Smart search</h3>
            <p>Find relevant information from your document context.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="surface-card">
            <div class="card-icon">💬</div>
            <h3>Natural chat</h3>
            <p>Ask questions in simple language and read clear answers.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="surface-card">
            <div class="card-icon">🛡️</div>
            <h3>Context-based</h3>
            <p>Answers are instructed to rely on retrieved PDF content.</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ==================================================
# INDEX PDF ONLY WHEN IT CHANGES
# ==================================================
file_bytes = uploaded_file.getvalue()

if (
    st.session_state.indexed_bytes != file_bytes
    or st.session_state.vectorstore is None
):
    try:
        with st.spinner("📚 Reading PDF and building your knowledge base..."):
            db, page_count, chunk_count = create_vectorstore(file_bytes)

        st.session_state.vectorstore = db
        st.session_state.indexed_bytes = file_bytes
        st.session_state.indexed_file = uploaded_file.name
        st.session_state.messages = []

        st.success(
            f"PDF ready! Processed {page_count} pages "
            f"into {chunk_count} text chunks."
        )

    except Exception:
        st.error(
            "Unable to read this PDF. Please upload a valid, "
            "text-readable PDF and try again."
        )
        st.stop()

else:
    st.markdown(
        f'<div class="status-note">✓ {st.session_state.indexed_file} '
        'is ready for questions.</div>',
        unsafe_allow_html=True
    )

db = st.session_state.vectorstore


# ==================================================
# CHAT HEADER
# ==================================================
st.markdown(
    '<div class="section-heading">💬 Chat with your document</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="section-subtitle">Ask a question and explore the '
    'information inside your PDF.</div>',
    unsafe_allow_html=True
)


# ==================================================
# CHAT HISTORY
# ==================================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(
                '<div class="answer-label">✦ DOCUMIND AI RESPONSE</div>',
                unsafe_allow_html=True
            )
        st.markdown(message["content"])


# ==================================================
# QUESTION INPUT AND ANSWERS
# ==================================================
query = st.chat_input("Ask anything about your PDF...")

if query:
    if not gemini_api_key:
        st.warning("Please enter your Gemini API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        st.markdown(
            '<div class="answer-label">✦ DOCUMIND AI RESPONSE</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Searching your document and preparing an answer..."):
            try:
                results = db.similarity_search(query, k=4)
                context = "\n\n".join(
                    doc.page_content for doc in results
                )

                llm = ChatGoogleGenerativeAI(
                    model="gemini-3.5-flash",
                    temperature=0,
                    google_api_key=gemini_api_key
                )

                prompt = f"""
You are a careful PDF question-answering assistant.

Rules:
- Answer using only the supplied context.
- If the answer is not available in the context, clearly say:
  "The provided context does not contain information regarding this question."
- Never invent names, dates, or facts.
- Use readable Markdown with short paragraphs, headings, or bullet points.
- Be direct and relevant to the question.

Context:
{context}

Question:
{query}

Answer:
"""

                response = llm.invoke(prompt)
                content = response.content

                if isinstance(content, list):
                    text_output = "\n".join(
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict)
                    )
                else:
                    text_output = str(content)

                if not text_output.strip():
                    text_output = (
                        "I could not generate an answer. "
                        "Please try asking the question again."
                    )

                st.markdown(text_output)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": text_output
                })

            except Exception:
                st.error(
                    "Unable to generate an answer. Check your API key, "
                    "model availability, internet connection, and try again."
                )


# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#8491A7;font-size:12px;">'
    'DocuMind AI · Built with Streamlit, FAISS, HuggingFace and Gemini'
    '</p>',
    unsafe_allow_html=True
)
