import streamlit as st
import json
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from llm_entity import TASK_LLM


@st.cache_resource
def load_model():
    return TASK_LLM  # Load LLM only once


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
