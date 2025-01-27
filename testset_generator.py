# import streamlit as st
# import json
# import pandas as pd
# from langchain_core.output_parsers import JsonOutputParser
# from langchain.prompts import PromptTemplate
# from llm_entity import HUG_LLM, TASK_LLM

# # Load the LLM model
# # huggingface_llm = HUG_LLM
# huggingface_llm = TASK_LLM
# st.set_page_config(page_title="Q&A Generator", page_icon="📝", layout="wide")

# def split_context(context, chunk_size=1000):
#     """Split context into chunks of specified size."""
#     return [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

# def generate_questions(llm, context, num_questions=5, question_complexity="simple"):
#     if not context or len(context) < 50:
#         st.warning("Context is too short to generate meaningful questions.")
#         return []

#     truncated_context = context[:2000]

#     prompt = """
# You are a test question generator. Based on the given context, create {num_questions} {question_complexity} questions and their answers.

# Context:
# {context}

# **Requirements**:
# - Generate {num_questions} {question_complexity} questions
# - Each question should have a clear, informative answer
# - Provide answers that are detailed and contextually rich
# - Output format must be a JSON list of objects

# **JSON Output Format**:
# [
#     {{
#         "question": "A clear, specific question from the context",
#         "groundtruth": "A comprehensive and detailed answer"
#     }},
#     ... (repeat for number of questions)
# ]
# """

#     try:
#         rephraser_system_prompt = PromptTemplate(
#             template=prompt,
#             input_variables=["context", "num_questions", "question_complexity"],
#         )

#         retrieval_rephraser = rephraser_system_prompt | llm | JsonOutputParser()

#         result = retrieval_rephraser.invoke(
#             {
#                 "context": truncated_context,
#                 "num_questions": num_questions,
#                 "question_complexity": question_complexity,
#             }
#         )

#         print(result)

#         if not isinstance(result, list):
#             st.error("Generated output is not in the expected format.")
#             return []

#         return result

#     except Exception as e:
#         st.error(f"Error generating questions: {e}")
#         return []

# # Initialize session state
# if 'results' not in st.session_state:
#     st.session_state.results = None
# if 'context' not in st.session_state:
#     st.session_state.context = ""

# # Streamlit App UI
# st.title("Q&A Generator")
# st.write("Generate questions and answers from text content.")

# # File uploader and text area options
# input_option = st.radio("Choose input method:", ["Upload File", "Paste Text"])

# if input_option == "Upload File":
#     uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
#     if uploaded_file:
#         st.session_state.context = uploaded_file.read().decode("utf-8")
# elif input_option == "Paste Text":
#     st.session_state.context = st.text_area("Paste your content here:")

# # New slider for chunk size
# chunk_size = st.slider("Context chunk size:", min_value=500, max_value=2000, value=1000)
# num_questions = st.slider("Number of questions per chunk:", min_value=1, max_value=10, value=5)
# question_complexity = st.selectbox("Question complexity:", ["simple", "multi-context", "reasoning"])

# if st.button("Generate Questions"):
#     if st.session_state.context.strip():
#         # Split context into chunks
#         context_chunks = split_context(st.session_state.context, chunk_size)

#         # Generate questions for each chunk
#         all_results = []
#         llm = huggingface_llm

#         for idx, chunk in enumerate(context_chunks, 1):
#             qa_pairs = generate_questions(llm, chunk, num_questions, question_complexity)

#             all_results.extend([
#                 {
#                     "chunk_number": idx,
#                     "chunk_text_reference": chunk,  # Preview of chunk
#                     "question": item.get("question", ""),
#                     "groundtruth": item.get("groundtruth", "")
#                 }
#                 for item in qa_pairs
#             ])

#         # Update session state with results
#         st.session_state.results = all_results

# # Always display results if they exist
# if st.session_state.results:
#     st.write("Generated Questions Preview:")
#     df = pd.DataFrame(st.session_state.results)
#     st.dataframe(df)

#     download_format = st.radio("Choose download format:", ["JSON", "CSV"])

#     if download_format == "JSON":
#         st.download_button(
#             "Download Q&A as JSON",
#             data=json.dumps(st.session_state.results, indent=2),
#             file_name="qa_test_set.json",
#             mime="application/json",
#         )
#     else:
#         st.download_button(
#             "Download Q&A as CSV",
#             data=df.to_csv(index=False).encode('utf-8'),
#             file_name="qa_test_set.csv",
#             mime="text/csv",
#         )


import streamlit as st
import json
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from llm_entity import HUG_LLM, TASK_LLM

# Load the LLM model
huggingface_llm = TASK_LLM

st.set_page_config(page_title="Q&A Generator", page_icon="📝", layout="wide")

# Enhanced UI layout
st.title("📝 AI-Powered Q&A Generator")
st.write("Generate questions and answers from your text content effortlessly.")

# Sidebar for input selection and settings
st.sidebar.header("Settings")

input_option = st.sidebar.radio("Choose input method:", ["Upload File", "Paste Text"])
question_format = st.sidebar.selectbox(
    "Select Question Type:",
    ["MCQs", "Open-ended", "True/False"],
)

chunk_size = st.sidebar.slider(
    "Context chunk size:", min_value=500, max_value=2000, value=1000
)
num_questions = st.sidebar.slider(
    "Number of questions per chunk:", min_value=1, max_value=10, value=5
)
question_complexity = st.sidebar.selectbox(
    "Question Complexity:",
    ["simple", "multi-context", "reasoning"],
)


# # Function to split context into chunks
# def split_context(context, chunk_size=1000):
#     """Split context into chunks of specified size."""
#     return [context[i : i + chunk_size] for i in range(0, len(context), chunk_size)]

def split_context(context, chunk_size=1000, overlap=300):
    """Split context into overlapping chunks."""
    chunks = []
    start = 0

    while start < len(context):
        end = start + chunk_size
        chunk = context[start:end]
        chunks.append(chunk)

        # Move the start position forward but keep an overlap
        start += chunk_size - overlap

    return chunks



# Function to generate questions based on type
def generate_questions(
    llm, context, num_questions, question_complexity, question_format
):
    if not context or len(context) < 50:
        st.warning("Context is too short to generate meaningful questions.")
        return []

    truncated_context = context[:2000]

    # Customize the prompt based on the selected question format
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


# File uploader or text input
st.sidebar.subheader("Input Your Content")
if input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
    if uploaded_file:
        st.session_state.context = uploaded_file.read().decode("utf-8")
elif input_option == "Paste Text":
    st.session_state.context = st.text_area("Paste your content here:")

if st.sidebar.button("Generate Questions"):
    if st.session_state.context.strip():
        context_chunks = split_context(st.session_state.context, chunk_size, overlap=300)
        all_results = []
        llm = huggingface_llm

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

# Display results
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

# Footer section
st.markdown(
    """
    ---
    **Powered by AI | Developed for automated Q&A generation**
    """,
    unsafe_allow_html=True,
)
