import pandas as pd
import numpy as np
import re
import streamlit as st
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

import contractions

from dotenv import load_dotenv
import os
from llm_entity import TASK_LLM
from langchain.prompts import PromptTemplate
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@st.cache_resource
def load_model():
    return TASK_LLM  # Load LLM only once


# Load environment variables
load_dotenv()
huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Download NLTK data
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")


# Function to preprocess text
def preprocess_text(text, steps):
    if "Convert to lowercase" in steps:
        text = text.lower()
    if "Remove punctuation" in steps:
        text = re.sub(r"[^\w\s]", "", text)
    if "Remove URLs" in steps:
        text = re.sub(r"http\S+", "", text)
    if "Remove Mentions" in steps:
        text = re.sub(r"@\w+", "", text)
    if "Remove Hashtags" in steps:
        text = re.sub(r"#\w+", "", text)
    if "Remove Non-Alphabetic" in steps:
        text = re.sub(r"[^a-zA-Z\s]", "", text)
    if "Expand Contractions" in steps:
        text = contractions.fix(text)
    if "Remove Emojis" in steps:
        text = text.encode("ascii", "ignore").decode("ascii")
    tokens = word_tokenize(text)
    if "Remove stopwords" in steps:
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]
    if "Lemmatize words" in steps:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = " ".join(tokens)
    if "Stem words" in steps:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    if "Remove Extra Whitespaces" in steps:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# Function to perform sentiment analysis
# def assign_sentiment(text, method="vader"):
#     if method == "vader":
#         analyzer = SentimentIntensityAnalyzer()
#         score = analyzer.polarity_scores(text)["compound"]
#         return (
#             1 if score > 0.05 else 0 if score < -0.05 else 0.5
#         )

#     elif method == "textblob":
#         score = TextBlob(text).sentiment.polarity
#         return 1 if score > 0 else 0 if score < 0 else 0.5
    
def assign_sentiment(text, method="vader"):
    if method == "vader":
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)["compound"]
        # Return 1 for positive, 0 for negative
        return 1 if score > 0.05 else 0

    elif method == "textblob":
        score = TextBlob(text).sentiment.polarity
        # Return 1 for positive, 0 for negative
        return 1 if score > 0 else 0


# Function to analyze results with LLM
def analyze_with_llm(results):
    """
    Sends the results to a Hugging Face LLM for analysis and feedback.
    """
    formatted_results = "\n".join(
        [
            f"Classifier: {name}\n"
            + "\n".join(
                [
                    (
                        f"{metric}: {value:.2f}"
                        if isinstance(value, (int, float))
                        else f"{metric}:\n{value}"
                    )
                    for metric, value in metrics.items()
                ]
            )
            for name, metrics in results.items()
        ]
    )

    # Define the prompt for the LLM
    prompt = f"""
    You are an expert in machine learning and model evaluation. 
    Please analyze the following classification results and provide insights:
    
    {formatted_results}
    
    Highlight any potential improvements or issues with the results.
    """

    prompt_template = PromptTemplate(
        template=prompt,
        input_variables=["formatted_results"],
    )
    question_generator = prompt_template | load_model()
    result = question_generator.invoke(
        {
            "context": formatted_results,
        }
    )
    return result.content

# Function to display the DataFrame with the selected columns
def write_to_df():
    if "sentiment" not in data.columns:
        # data["sentiment"] = data[column_to_preprocess].apply(
        #     lambda x: assign_sentiment(x, selected_classifiers[0])
        st.dataframe(
                data[[column_to_preprocess, "preprocessed_text"]]
            )  # Display three columns
    else:
        st.dataframe(
            data[[column_to_preprocess, "preprocessed_text", "sentiment"]]
        )


# Sidebar for inputs
st.sidebar.title("Build Your Sentiment Classification Pipeline")
# uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# # Column selection for preprocessing
# column_to_preprocess = None
# isCustomAnalyzer = False

# # Check if a file is uploaded
# if uploaded_file:
#     st.write("### Dataset Uploaded")
#     print("begining here")
#     data = pd.read_csv(uploaded_file)
#     if "sentiment" not in data.columns:
#         st.warning("The CSV file must contain a 'sentiment' column.")
#         isCustomAnalyzer = True
#     else:
#         isCustomAnalyzer = False
# At the start of your script, define all the session state variables that need to be reset
def reset_session_state():
    session_states = [
        "uploaded_data",
        "preprocessed_data",
        "sentiment_data",
        "preprocessing_steps_selected",
        "selected_analyzer",
        "results"
    ]
    for state in session_states:
        if state in st.session_state:
            del st.session_state[state]

# Modify your file uploader section
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Check if a file is uploaded
if uploaded_file:
    # Get the name of the currently uploaded file
    file_name = uploaded_file.name
    
    # Check if this is a new file
    if "current_file" not in st.session_state or st.session_state["current_file"] != file_name:
        # Reset everything if it's a new file
        reset_session_state()
        st.session_state["current_file"] = file_name
    
    st.write("### Dataset Uploaded")
    data = pd.read_csv(uploaded_file)
    if "sentiment" not in data.columns:
        st.warning("The CSV file must contain a 'sentiment' column.")
        isCustomAnalyzer = True
    else:
        isCustomAnalyzer = False

    st.sidebar.write("### Select the Column to Preprocess")
    column_to_preprocess = st.sidebar.selectbox(
        "Select the column to preprocess:", data.columns
    )

# Preprocessing options in the sidebar
preprocessing_steps = st.sidebar.multiselect(
    "Choose preprocessing steps:",
    [
        "Convert to lowercase",
        "Remove punctuation",
        "Remove URLs",
        "Remove Mentions",
        "Remove Hashtags",
        "Remove Non-Alphabetic",
        "Expand Contractions",
        "Remove Emojis",
        "Remove stopwords",
        "Lemmatize words",
        "Stem words",
        "Remove Extra Whitespaces",
    ],
)

# Preprocessing button in the sidebar
preprocess_btn = st.sidebar.button("Apply Preprocessing")

# Custom analyzer options in the sidebar
if isCustomAnalyzer:
    st.sidebar.write("### Generate Sentiment with Custom Analyzer")
    selected_analyzer = st.sidebar.selectbox(
        "Choose analyzer:",
        [
            "vader",
            "textblob",
        ],
    )

    # Custom analyzer button in the sidebar
    custom_analyzer_btn = st.sidebar.button("Generate Sentiment")

# Classifier and metrics options in the sidebar
st.sidebar.write("### Select Classifiers and Metrics")
selected_classifiers = st.sidebar.multiselect(
    "Choose classifiers:",
    [
        "Logistic Regression",
        "Naive Bayes",
        "Support Vector Machine",
        "Random Forest",
        "Gradient Boosting",
    ],
)

# Metrics selection in the sidebar
metrics = st.sidebar.multiselect(
    "Choose metrics to evaluate your model:",
    ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"],
)

# Run pipeline button in the sidebar
pipeline_btn = st.sidebar.button("Run Pipeline")

# Initialize session state if not already initialized
print("reached here")
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None
if "preprocessed_data" not in st.session_state:
    st.session_state["preprocessed_data"] = None
if "sentiment_data" not in st.session_state:
    st.session_state["sentiment_data"] = None
if "preprocessing_steps_selected" not in st.session_state:
    st.session_state["preprocessing_steps_selected"] = ""
if "selected_analyzer" not in st.session_state:
    st.session_state["selected_analyzer"] = ""
if "results" not in st.session_state:
    st.session_state["results"] = {}


# Only proceed if a file is uploaded
if uploaded_file:
    # Load dataset
    st.session_state["uploaded_data"] = data  # Store uploaded data in session state
    st.write("### Dataset Preview")
    st.write(data.head())

    # if "sentiment" not in data.columns:
    #     st.warning("The CSV file must contain a 'sentiment' column.")
    # else:
    # Main content area
    st.title("Sentiment Classification Pipeline")

    # Display the preprocessing table if preprocessed data exists
    if st.session_state["preprocessed_data"] is not None:
        # Add preprocessed data as a new column in the DataFrame
        data["preprocessed_text"] = st.session_state["preprocessed_data"]
        st.write("### Preprocessed Data Preview")
        write_to_df()

    if st.session_state["sentiment_data"] is not None:
        # Add preprocessed data as a new column in the DataFrame
        data["sentiment"] = st.session_state["sentiment_data"]
        st.write("### Sentiment Data Preview")
        write_to_df()

    # Apply preprocessing when button clicked
    if preprocess_btn:
        with st.spinner("Preprocessing... Please wait."):
            if not preprocessing_steps:
                st.warning(
                    "No preprocessing steps selected. The raw text will be used."
                )
                st.session_state["preprocessed_data"] = data[column_to_preprocess]
            else:
                st.write("### Preprocessing Data...")
                # Apply preprocessing and store as a new column
                st.session_state["preprocessed_data"] = data[
                    column_to_preprocess
                ].apply(lambda x: preprocess_text(x, preprocessing_steps))
            # Display the new column in the data
            data["preprocessed_text"] = st.session_state["preprocessed_data"]
            write_to_df()

    # Generate sentiment with custom analyzer when button clicked
    if isCustomAnalyzer:
        if custom_analyzer_btn: 
            # Check if the button is clicked
            with st.spinner("Generating sentiment... Please wait."):
                if not selected_analyzer:
                    st.warning("No analyzer selected. Please select at least one analyzer.")
                else:
                    st.write("### Generating Sentiment with Custom Analyzer...")
                    st.session_state["sentiment_data"] = data["preprocessed_text"].apply(
                        lambda x: assign_sentiment(x, selected_analyzer)
                    )
                    # Add sentiment as a new column in the DataFrame
                    data["sentiment"] = st.session_state["sentiment_data"]
                    st.session_state["uploaded_data"] = data  # Store updated data in session state
                    write_to_df()




    if "sentiment" in data.columns:
        # Ensure preprocessed data is available before using it in train_test_split
        if st.session_state["preprocessed_data"] is not None:
            X = st.session_state["preprocessed_data"]
            y = data["sentiment"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if pipeline_btn:
                with st.spinner("Training and evaluating models... Please wait."):
                    if not selected_classifiers:
                        st.warning(
                            "No classifiers selected. Please select at least one classifier."
                        )
                    elif not metrics:
                        st.warning(
                            "No metrics selected. Please select at least one evaluation metric."
                        )
                    else:
                        st.write("### Training and Evaluating Models...")
                        classifiers = {
                            "Logistic Regression": LogisticRegression(),
                            "Naive Bayes": MultinomialNB(),
                            "Support Vector Machine": SVC(),
                            "Random Forest": RandomForestClassifier(),
                            "Gradient Boosting": GradientBoostingClassifier(),
                        }
                        results = {}  # Initialize results
                        y_preds = {}

                        # Train and predict for each selected classifier
                        for name in selected_classifiers:
                            classifier = classifiers[name]
                            pipeline = Pipeline(
                                [("tfidf", TfidfVectorizer()), ("clf", classifier)]
                            )
                            pipeline.fit(X_train, y_train)
                            y_pred = pipeline.predict(X_test)

                            # Store the predictions for later use in confusion matrix plotting
                            y_preds[name] = y_pred

                            # Calculate metrics
                            model_metrics = {}
                            if "Accuracy" in metrics:
                                model_metrics["Accuracy"] = accuracy_score(
                                    y_test, y_pred
                                )
                            if "Precision" in metrics:
                                model_metrics["Precision"] = precision_score(
                                    y_test, y_pred, average="weighted"
                                )
                            if "Recall" in metrics:
                                model_metrics["Recall"] = recall_score(
                                    y_test, y_pred, average="weighted"
                                )
                            if "F1 Score" in metrics:
                                model_metrics["F1 Score"] = f1_score(
                                    y_test, y_pred, average="weighted"
                                )

                            # Include confusion matrix in results if selected
                            if "Confusion Matrix" in metrics:
                                cm = confusion_matrix(y_test, y_pred)
                                model_metrics["Confusion Matrix"] = cm.tolist()

                            results[name] = model_metrics

                        # Save results in session state
                        st.session_state["results"] = results

                        # Display results
                        st.write(f"## Metrics Results")
                        for name, model_metrics in results.items():
                            st.write(f"### {name}")
                            for metric_name, metric_value in model_metrics.items():
                                if metric_name != "Confusion Matrix":
                                    st.write(
                                        f"**{metric_name}:** {metric_value:.2f}"
                                    )  # Only format numerical values

                        # Plot confusion matrices for each selected classifier
                        if "Confusion Matrix" in metrics:
                            st.write(f"## Confusion Matrix Plot")
                            for name, y_pred in y_preds.items():
                                cm = confusion_matrix(y_test, y_pred)
                                disp = ConfusionMatrixDisplay(
                                    confusion_matrix=cm,
                                    display_labels=["Negative", "Positive"],
                                )
                                disp.plot(cmap=plt.cm.Blues)
                                plt.title(f"Confusion Matrix: {name}")
                                st.pyplot(plt.gcf())
                                plt.clf()  # Clear the current figure to avoid overlap in next plot
                        # # Plot confusion matrices for each selected classifier
                        # if "Confusion Matrix" in metrics:
                        #     st.write(f"## Confusion Matrix Plot")
                        #     for name, y_pred in y_preds.items():
                        #         cm = confusion_matrix(y_test, y_pred)
                        #         # Get unique labels from the data
                        #         unique_labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
                                
                        #         disp = ConfusionMatrixDisplay(
                        #             confusion_matrix=cm,
                        #             display_labels=unique_labels
                        #         )
                        #         fig, ax = plt.subplots()
                        #         disp.plot(ax=ax, cmap=plt.cm.Blues)
                        #         plt.title(f"Confusion Matrix: {name}")
                        #         st.pyplot(fig)
                        #         plt.close(fig)


# Add the button to the sidebar
if st.sidebar.button("Analyze Results with LLM"):
    # Check if results are available in session state
    if "results" in st.session_state and st.session_state["results"]:
        with st.spinner("Analyzing results with LLM..."):
            try:
                llm_analysis = analyze_with_llm(st.session_state["results"])
                st.write("### LLM Analysis of Results")
                st.write(llm_analysis)
            except Exception as e:
                st.error(f"An error occurred while analyzing with LLM: {e}")
    else:
        st.warning("No results available to analyze. Run the pipeline first.")
