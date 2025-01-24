# import streamlit as st
# import json
# import pandas as pd
# from langchain_core.output_parsers import JsonOutputParser
# from langchain.prompts import PromptTemplate
# from llm_entity import get_llm, TASK_LLM

# # Load the LLM model
# huggingface_llm = TASK_LLM
# st.set_page_config(page_title="Q&A Generator", page_icon="📝", layout="wide")

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

# num_questions = st.slider("Number of questions:", min_value=1, max_value=100, value=5)
# question_complexity = st.selectbox("Question complexity:", ["simple", "complex"])

# if st.button("Generate Questions"):
#     if st.session_state.context.strip():
#         llm = huggingface_llm
#         qa_pairs = generate_questions(llm, st.session_state.context, num_questions, question_complexity)

#         if qa_pairs:
#             st.session_state.results = [
#                 {
#                     "question": item.get("question", ""),
#                     "groundtruth": item.get("groundtruth", ""),
#                     "context": st.session_state.context[:1000]
#                 }
#                 for item in qa_pairs
#             ]

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
from llm_entity import get_llm, TASK_LLM

# Load the LLM model
huggingface_llm = TASK_LLM
st.set_page_config(page_title="Q&A Generator", page_icon="📝", layout="wide")

def split_context(context, chunk_size=1000):
    """Split context into chunks of specified size."""
    return [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]

def generate_questions(llm, context, num_questions=5, question_complexity="simple"):
    if not context or len(context) < 50:
        st.warning("Context is too short to generate meaningful questions.")
        return []

    truncated_context = context[:2000]

    prompt = """
You are a test question generator. Based on the given context, create {num_questions} {question_complexity} questions and their answers.

Context:
{context}

**Requirements**:
- Generate {num_questions} {question_complexity} questions
- Each question should have a clear, informative answer
- Provide answers that are detailed and contextually rich
- Output format must be a JSON list of objects

**JSON Output Format**:
[
    {{
        "question": "A clear, specific question from the context",
        "groundtruth": "A comprehensive and detailed answer"
    }},
    ... (repeat for number of questions)
]
"""

    try:
        rephraser_system_prompt = PromptTemplate(
            template=prompt,
            input_variables=["context", "num_questions", "question_complexity"],
        )

        retrieval_rephraser = rephraser_system_prompt | llm | JsonOutputParser()

        result = retrieval_rephraser.invoke(
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

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'context' not in st.session_state:
    st.session_state.context = ""

# Streamlit App UI
st.title("Q&A Generator")
st.write("Generate questions and answers from text content.")

# File uploader and text area options
input_option = st.radio("Choose input method:", ["Upload File", "Paste Text"])

if input_option == "Upload File":
    uploaded_file = st.file_uploader("Upload a text file:", type=["txt"])
    if uploaded_file:
        st.session_state.context = uploaded_file.read().decode("utf-8")
elif input_option == "Paste Text":
    st.session_state.context = st.text_area("Paste your content here:")

# New slider for chunk size
chunk_size = st.slider("Context chunk size:", min_value=500, max_value=2000, value=1000)
num_questions = st.slider("Number of questions per chunk:", min_value=1, max_value=10, value=5)
question_complexity = st.selectbox("Question complexity:", ["simple", "complex"])

if st.button("Generate Questions"):
    if st.session_state.context.strip():
        # Split context into chunks
        context_chunks = split_context(st.session_state.context, chunk_size)
        
        # Generate questions for each chunk
        all_results = []
        llm = huggingface_llm
        
        for idx, chunk in enumerate(context_chunks, 1):
            qa_pairs = generate_questions(llm, chunk, num_questions, question_complexity)
            
            all_results.extend([
                {
                    "chunk_number": idx,
                    "chunk_text": chunk[:200] + "...",  # Preview of chunk
                    "question": item.get("question", ""),
                    "groundtruth": item.get("groundtruth", "")
                }
                for item in qa_pairs
            ])

        # Update session state with results
        st.session_state.results = all_results

# Always display results if they exist
if st.session_state.results:
    st.write("Generated Questions Preview:")
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
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="qa_test_set.csv",
            mime="text/csv",
        )