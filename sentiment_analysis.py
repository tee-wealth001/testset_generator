# import pandas as pd
# import re
# import streamlit as st
# import nltk
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     ConfusionMatrixDisplay,
# )
# import matplotlib.pyplot as plt
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import contractions

# # Download NLTK data
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")


# # Function to preprocess text
# def preprocess_text(text, steps):
#     if "Convert to lowercase" in steps:
#         text = text.lower()
#     if "Remove punctuation" in steps:
#         text = re.sub(r"[^\w\s]", "", text)
#     if "Remove URLs" in steps:
#         text = re.sub(r"http\S+", "", text)
#     if "Remove Mentions" in steps:
#         text = re.sub(r"@\w+", "", text)
#     if "Remove Hashtags" in steps:
#         text = re.sub(r"#\w+", "", text)
#     if "Remove Non-Alphabetic" in steps:
#         text = re.sub(r"[^a-zA-Z\s]", "", text)
#     if "Expand Contractions" in steps:
#         text = contractions.fix(text)
#     if "Remove Emojis" in steps:
#         text = text.encode("ascii", "ignore").decode("ascii")
#     tokens = word_tokenize(text)
#     if "Remove stopwords" in steps:
#         stop_words = set(stopwords.words("english"))
#         tokens = [word for word in tokens if word not in stop_words]
#     if "Lemmatize words" in steps:
#         lemmatizer = WordNetLemmatizer()
#         tokens = [lemmatizer.lemmatize(word) for word in tokens]
#     text = " ".join(tokens)
#     if "Remove Extra Whitespaces" in steps:
#         text = re.sub(r"\s+", " ", text).strip()
#     return text


# # Sidebar for inputs
# st.sidebar.title("Build Your Sentiment Classification Pipeline")
# uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# # Preprocessing options in the sidebar
# preprocessing_steps = st.sidebar.multiselect(
#     "Choose preprocessing steps:",
#     [
#         "Convert to lowercase",
#         "Remove punctuation",
#         "Remove URLs",
#         "Remove Mentions",
#         "Remove Hashtags",
#         "Remove Non-Alphabetic",
#         "Expand Contractions",
#         "Remove Emojis",
#         "Remove stopwords",
#         "Lemmatize words",
#         "Remove Extra Whitespaces",
#     ],
# )

# # Classifier and metrics options in the sidebar
# st.sidebar.write("### Select Classifiers and Metrics")
# selected_classifiers = st.sidebar.multiselect(
#     "Choose classifiers:",
#     [
#         "Logistic Regression",
#         "Naive Bayes",
#         "Support Vector Machine",
#         "Random Forest",
#         "Gradient Boosting",
#     ],
# )
# metrics = st.sidebar.multiselect(
#     "Choose metrics to evaluate your model:",
#     ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"],
# )

# # Initialize session state if not already initialized
# if "uploaded_data" not in st.session_state:
#     st.session_state["uploaded_data"] = None
# if "preprocessed_data" not in st.session_state:
#     st.session_state["preprocessed_data"] = None
# if "preprocessing_steps_selected" not in st.session_state:
#     st.session_state["preprocessing_steps_selected"] = []

# # Only proceed if a file is uploaded
# if uploaded_file:
#     # Load dataset
#     data = pd.read_csv(uploaded_file)
#     st.session_state["uploaded_data"] = data  # Store uploaded data in session state
#     st.write("### Dataset Preview")
#     st.write(data.head())

#     if "tweet" not in data.columns or "sentiment" not in data.columns:
#         st.error("The CSV file must contain 'tweet' and 'sentiment' columns.")
#     else:
#         # Main content area
#         st.title("Sentiment Classification Pipeline")

#         # Display the preprocessing table if preprocessed data exists
#         if st.session_state["preprocessed_data"] is not None:
#             # Add preprocessed data as a new column in the DataFrame
#             data["preprocessed_tweet"] = st.session_state["preprocessed_data"]
#             st.write("### Preprocessed Data Preview")
#             st.dataframe(
#                 data[["tweet", "preprocessed_tweet", "sentiment"]]
#             )  # Display three columns

#         # Apply preprocessing when button clicked
#         if st.button("Apply Preprocessing"):
#             with st.spinner("Preprocessing... Please wait."):
#                 if not preprocessing_steps:
#                     st.warning(
#                         "No preprocessing steps selected. The raw text will be used."
#                     )
#                     st.session_state["preprocessed_data"] = data["tweet"]
#                 else:
#                     st.write("### Preprocessing Data...")
#                     # Apply preprocessing and store as a new column
#                     st.session_state["preprocessed_data"] = data["tweet"].apply(
#                         lambda x: preprocess_text(x, preprocessing_steps)
#                     )
#                 # Display the new column in the data
#                 data["preprocessed_tweet"] = st.session_state["preprocessed_data"]
#                 st.write(
#                     data[["tweet", "preprocessed_tweet", "sentiment"]]
#                 )  # Display the new dataframe with 3 columns

#         # Ensure preprocessed data is available before using it in train_test_split
#         if st.session_state["preprocessed_data"] is not None:
#             X = st.session_state["preprocessed_data"]
#             y = data["sentiment"]
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X, y, test_size=0.2, random_state=42
#             )

#             # Train and evaluate models
#             # Train and evaluate models
#             if st.button("Run Pipeline"):
#                 with st.spinner("Training and evaluating models... Please wait."):
#                     if not selected_classifiers:
#                         st.warning(
#                             "No classifiers selected. Please select at least one classifier."
#                         )
#                     elif not metrics:
#                         st.warning(
#                             "No metrics selected. Please select at least one evaluation metric."
#                         )
#                     else:
#                         st.write("### Training and Evaluating Models...")
#                         classifiers = {
#                             "Logistic Regression": LogisticRegression(),
#                             "Naive Bayes": MultinomialNB(),
#                             "Support Vector Machine": SVC(),
#                             "Random Forest": RandomForestClassifier(),
#                             "Gradient Boosting": GradientBoostingClassifier(),
#                         }
#                         results = {}
#                         y_preds = {}

#                         # Train and predict for each selected classifier
#                         for name in selected_classifiers:
#                             classifier = classifiers[name]
#                             pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", classifier)])
#                             pipeline.fit(X_train, y_train)
#                             y_pred = pipeline.predict(X_test)

#                             # Store the predictions for later use in confusion matrix plotting
#                             y_preds[name] = y_pred

#                             # Calculate metrics
#                             model_metrics = {}
#                             if "Accuracy" in metrics:
#                                 model_metrics["Accuracy"] = accuracy_score(y_test, y_pred)
#                             if "Precision" in metrics:
#                                 model_metrics["Precision"] = precision_score(
#                                     y_test, y_pred, average="weighted"
#                                 )
#                             if "Recall" in metrics:
#                                 model_metrics["Recall"] = recall_score(
#                                     y_test, y_pred, average="weighted"
#                                 )
#                             if "F1 Score" in metrics:
#                                 model_metrics["F1 Score"] = f1_score(
#                                     y_test, y_pred, average="weighted"
#                                 )

#                             results[name] = model_metrics

#                         # Display results
#                         st.write(f"## Metrics Results")
#                         for name, model_metrics in results.items():
#                             st.write(f"## {name}")
#                             for metric_name, metric_value in model_metrics.items():
#                                 st.write(f"**{metric_name}:** {metric_value:.2f}")

#                         # Plot confusion matrices for each selected classifier
#                         st.write(f"## Confusion Matrix Plot")
#                         for name, y_pred in y_preds.items():
#                             cm = confusion_matrix(y_test, y_pred)
#                             disp = ConfusionMatrixDisplay(
#                                 confusion_matrix=cm, display_labels=["Negative", "Positive"]
#                             )
#                             disp.plot(cmap=plt.cm.Blues)
#                             plt.title(f"Confusion Matrix: {name}")
#                             st.pyplot(plt.gcf())
#                             plt.clf()  # Clear the current figure to avoid overlap in next plot


import pandas as pd
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
import contractions

# Download NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


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
    if "Remove Extra Whitespaces" in steps:
        text = re.sub(r"\s+", " ", text).strip()
    return text


# Sidebar for inputs
st.sidebar.title("Build Your Sentiment Classification Pipeline")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Column selection for preprocessing
column_to_preprocess = None
if uploaded_file:
    data = pd.read_csv(uploaded_file)
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
        "Remove Extra Whitespaces",
    ],
)

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
metrics = st.sidebar.multiselect(
    "Choose metrics to evaluate your model:",
    ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"],
)

# Initialize session state if not already initialized
if "uploaded_data" not in st.session_state:
    st.session_state["uploaded_data"] = None
if "preprocessed_data" not in st.session_state:
    st.session_state["preprocessed_data"] = None
if "preprocessing_steps_selected" not in st.session_state:
    st.session_state["preprocessing_steps_selected"] = []

# Only proceed if a file is uploaded
if uploaded_file:
    # Load dataset
    st.session_state["uploaded_data"] = data  # Store uploaded data in session state
    st.write("### Dataset Preview")
    st.write(data.head())

    if "sentiment" not in data.columns:
        st.error("The CSV file must contain a 'sentiment' column.")
    else:
        # Main content area
        st.title("Sentiment Classification Pipeline")

        # Display the preprocessing table if preprocessed data exists
        if st.session_state["preprocessed_data"] is not None:
            # Add preprocessed data as a new column in the DataFrame
            data["preprocessed_text"] = st.session_state["preprocessed_data"]
            st.write("### Preprocessed Data Preview")
            st.dataframe(
                data[[column_to_preprocess, "preprocessed_text", "sentiment"]]
            )  # Display three columns

        # Apply preprocessing when button clicked
        if st.button("Apply Preprocessing"):
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
                st.write(
                    data[[column_to_preprocess, "preprocessed_text", "sentiment"]]
                )  # Display the new dataframe with 3 columns

        # Ensure preprocessed data is available before using it in train_test_split
        if st.session_state["preprocessed_data"] is not None:
            X = st.session_state["preprocessed_data"]
            y = data["sentiment"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train and evaluate models
            if st.button("Run Pipeline"):
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
                        results = {}
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

                            results[name] = model_metrics

                        # Display results
                        # st.write(f"## Metrics Results")
                        # for name, model_metrics in results.items():
                        st.write(f"## Metrics Results")
                        for name, model_metrics in results.items():
                            st.write(f"## {name}")
                            for metric_name, metric_value in model_metrics.items():
                                st.write(f"**{metric_name}:** {metric_value:.2f}")

                        # Plot confusion matrices for each selected classifier
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
