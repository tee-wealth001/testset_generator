import streamlit as st
import json
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from llm_entity import TASK_LLM


@st.cache_resource
def load_model():
    return TASK_LLM  # Load LLM only once

#Css styles
def load_css():
    st.markdown(
        """
<style>
/* Global Styles */


/* Title and Header Styles */
.stTitle {
    color: #4CAF50 !important;
    font-size: 3rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

h1 {
    color: #4CAF50 !important;
}

/* Sidebar Styles */
.css-1d391kg {
    background-color: #2d2d2d;
    padding: 2rem 1rem;
    border-right: 1px solid #3d3d3d;
}

.sidebar .stButton > button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 0.75rem;
    border-radius: 5px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.sidebar .stButton > button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Job Card Styles */
.job-card {
    background-color: #2d2d2d;
    border-radius: 8px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.job-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}

/* Button Styles */
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background-color: #45a049;
    transform: translateY(-2px);
}

/* Analysis Panel Styles */
.fixed-analysis {
    border-left: 1px solid #3d3d3d !important;
    padding: 2rem !important;
    box-shadow: -4px 0 8px rgba(0,0,0,0.1);
}

.fixed-analysis h3 {
    color: #4CAF50;
    margin-bottom: 1rem;
}

/* File Uploader Styles */
.stFileUploader {
    border-radius: 8px;
    padding: 1rem;
    border: 2px dashed #4CAF50;
}

/* Pagination Styles */
.pagination {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin: 2rem 0;
}

.pagination button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
}

/* Link Button Styles */
.stLinkButton > button {
    background-color: #2196F3;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 5px;
    font-weight: 500;
    transition: all 0.3s ease;
}

.stLinkButton > button:hover {
    background-color: #1976D2;
    transform: translateY(-2px);
}

/* Loading Spinner Styles */
.stSpinner {
    color: #4CAF50 !important;
}

/* Alert/Warning Styles */
.stAlert {
    border-radius: 5px;
    border-left: 4px solid #0abb59;
}

/* Success Message Styles */
.success-message {
    background-color: rgba(76, 175, 80, 0.1);
    border-left: 4px solid #4CAF50;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

/* Error Message Styles */
.error-message {
    background-color: rgba(244, 67, 54, 0.1);
    border-left: 4px solid #f44336;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}

/* Caption Styles */
.stCaption {
    color: #999;
    font-size: 0.9rem;
}

/* Divider Styles */
.stDivider {
    border-color: #3d3d3d;
    margin: 2rem 0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .fixed-analysis {
        position: relative !important;
        width: 100% !important;
        margin-top: 2rem;
    }
    
    .stTitle {
        font-size: 2rem;
    }
}
</style>
    """,
        unsafe_allow_html=True,
    )




st.set_page_config(page_title="Synthetic Testset Generator", page_icon="📝", layout="wide")

st.title("📝 AI-Powered Synthetic Testset Generator")
st.write("Generate questions and answers from your text content effortlessly.")

st.sidebar.header("Settings")

input_option = st.sidebar.radio("Choose input method:", ["Upload File", "Paste Text"])
question_format = st.sidebar.selectbox(
    "Select Question Type:", ["MCQs", "Open-ended", "True/False"]
)
chunk_size = st.sidebar.slider(
    "Context chunk size:", min_value=500, max_value=2000, value=1000
)
num_questions = st.sidebar.slider(
    "Number of questions per chunk:", min_value=1, max_value=10, value=5
)
question_complexity = st.sidebar.selectbox(
    "Question Complexity:", ["simple", "multi-context", "reasoning"]
)


def split_context(context, chunk_size=1000, overlap=300):
    chunks = []
    start = 0
    while start < len(context):
        end = start + chunk_size
        chunk = context[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def generate_questions(
    llm, context, num_questions, question_complexity, question_format
):
    if not context or len(context) < 50:
        st.warning("Context is too short to generate meaningful questions.")
        return []

    truncated_context = context[:2000]
    format_instructions = {
        "MCQs": """
Each question should have 4 options with one correct answer.

**JSON Output Format:**
[
    {{
        "question": "MCQ question from the context",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "correct_option": "Option B"
    }}
]
""",
        "Open-ended": """
Each question should have a detailed open-ended answer.

**JSON Output Format:**
[
    {{
        "question": "Open-ended question from the context",
        "groundtruth": "A short and precise answer"
    }}
]
""",
        "True/False": """
Each question should be in True/False format.

**JSON Output Format:**
[
    {{
        "question": "True/False question from the context",
        "answer": "True"
    }}
]
""",
    }

    prompt = f"""
You are a test question generator. Based on the given context, create {num_questions} {question_complexity} questions in {question_format} format.

Context:
{context}

**Requirements**:
- Generate {num_questions} {question_complexity} questions in {question_format} format
- Ensure accuracy and relevance to the context
{format_instructions[question_format]}
"""

    try:
        prompt_template = PromptTemplate(
            template=prompt,
            input_variables=["context", "num_questions", "question_complexity"],
        )
        question_generator = prompt_template | llm | JsonOutputParser()
        result = question_generator.invoke(
            {
                "context": truncated_context,
                "num_questions": num_questions,
                "question_complexity": question_complexity,
            }
        )
        if not isinstance(result, list):
            st.error("Generated output is not in the expected format.")
            return []
        return result
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []


st.sidebar.subheader("Input Your Content")
if input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
    if uploaded_file:
        st.session_state.context = uploaded_file.read().decode("utf-8")
elif input_option == "Paste Text":
    st.session_state.context = st.text_area("Paste your content here:")

if st.sidebar.button("Generate Questions"):
    with st.spinner("Generating questions, please wait..."):
        if st.session_state.context.strip():
            context_chunks = split_context(
                st.session_state.context, chunk_size, overlap=300
            )
            all_results = []
            llm = load_model()
            for idx, chunk in enumerate(context_chunks, 1):
                qa_pairs = generate_questions(
                    llm, chunk, num_questions, question_complexity, question_format
                )
                all_results.extend(
                    [
                        {
                            "chunk_number": idx,
                            "chunk_text_reference": chunk,
                            "question": item.get("question", ""),
                            "answer": (
                                item.get("groundtruth", "")
                                if question_format != "MCQs"
                                else item.get("correct_option", "")
                            ),
                            "options": (
                                item.get("options", [])
                                if question_format == "MCQs"
                                else None
                            ),
                        }
                        for item in qa_pairs
                    ]
                )
            st.session_state.results = all_results

if st.session_state.get("results"):
    st.subheader("Generated Questions Preview")
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df)
    download_format = st.radio("Choose download format:", ["JSON", "CSV"])
    if download_format == "JSON":
        st.download_button(
            "Download Q&A as JSON",
            data=json.dumps(st.session_state.results, indent=2),
            file_name="qa_test_set.json",
            mime="application/json",
        )
    else:
        st.download_button(
            "Download Q&A as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="qa_test_set.csv",
            mime="text/csv",
        )

st.markdown(
    """
    ---
    **Powered by AI | Developed for automated Q&A generation**
    """,
    unsafe_allow_html=True,
)

load_css()
