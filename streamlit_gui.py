import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from wordcloud import WordCloud
import pickle
import json


# Load your saved models
@st.cache_resource
def load_models():
    # Load the RoBERTa model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaForSequenceClassification.from_pretrained("./best_roberta_sentiment")
    roberta_model = RobertaForSequenceClassification.from_pretrained("./best_roberta_sentiment")

    from transformers import BertTokenizer, BertForSequenceClassification
    bert_tokenizer = BertTokenizer.from_pretrained("./best_bert_sentiment")
    bert_model = BertForSequenceClassification.from_pretrained("./best_bert_sentiment")

    # Load DistilBERT model
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    distilbert_tokenizer = DistilBertTokenizer.from_pretrained("./distilbert_sentiment")
    distilbert_model = DistilBertForSequenceClassification.from_pretrained("./distilbert_sentiment")

    # comparison_df = pd.read_csv("model_metrics.csv")
    # detailed_metrics = pd.read_csv("detailed_metrics.csv")
    # sentiment_counts = pd.read_csv("sentiment_distribution.csv")

    # Load traditional ML models
    tfidf_lr_pipeline = pickle.load(open("tfidf_lr_model.pkl", "rb"))
    # bow_nb_pipeline = pickle.load(open("bow_nb_model.pkl", "rb"))

    # with open("confusion_matrices.json", "r") as f:
    #     confusion_matrices = json.load(f)
    #     for model in confusion_matrices:
    #         confusion_matrices[model] = np.array(confusion_matrices[model])
    #
    # with open("feature_importance.json", "r") as f:
    #     feature_importance = json.load(f)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    return {
        "tokenizer": tokenizer,
        "roberta_model": roberta_model,
        "bert_tokenizer": bert_tokenizer,
        "bert_model": bert_model,
        "distilbert_tokenizer": distilbert_tokenizer,
        "distilbert_model": distilbert_model,
        "tfidf_lr": tfidf_lr_pipeline,
        # "bow_nb": bow_nb_pipeline,
        "lemmatizer": lemmatizer,
        "stop_words": stop_words,
        # "comparison_df": comparison_df,
        # "detailed_metrics": detailed_metrics,
        # "sentiment_counts": sentiment_counts,
        # "confusion_matrices": confusion_matrices,
        # "feature_importance": feature_importance

    }


# Load sample data for visualizations
@st.cache_data
def load_sample_data():
    df_sample = pd.read_csv("car_reviews_sample.csv")
    return df_sample


# Text preprocessing
def clean_text(text, lemmatizer, stop_words):
    if not isinstance(text, str):
        return ""

    # Convert to lowercase and remove HTML tags
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)


# Analyze sentiment using different models
def analyze_sentiment(text, models):
    cleaned_text = clean_text(text, models["lemmatizer"], models["stop_words"])

    results = {}

    # RoBERTa prediction
    # RoBERTa prediction
    inputs = models["tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = models["roberta_model"](**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)
        scores = predictions[0].tolist()
        roberta_pred = predictions[0].argmax().item()

    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    results["RoBERTa"] = {
        "prediction": sentiment_map[roberta_pred],
        "confidence": scores[roberta_pred],
        "all_scores": {sentiment_map[i]: scores[i] for i in range(3)}
    }

    # BERT prediction
    bert_inputs = models["bert_tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        bert_outputs = models["bert_model"](**bert_inputs)
        bert_predictions = torch.softmax(bert_outputs.logits, dim=1)
        bert_scores = bert_predictions[0].tolist()
        bert_pred = bert_predictions[0].argmax().item()

    results["BERT"] = {
        "prediction": sentiment_map[bert_pred],
        "confidence": bert_scores[bert_pred],
        "all_scores": {sentiment_map[i]: bert_scores[i] for i in range(3)}
    }

    # DistilBERT prediction
    distilbert_inputs = models["distilbert_tokenizer"](text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        distilbert_outputs = models["distilbert_model"](**distilbert_inputs)
        distilbert_predictions = torch.softmax(distilbert_outputs.logits, dim=1)
        distilbert_scores = distilbert_predictions[0].tolist()
        distilbert_pred = distilbert_predictions[0].argmax().item()

    results["DistilBERT"] = {
        "prediction": sentiment_map[distilbert_pred],
        "confidence": distilbert_scores[distilbert_pred],
        "all_scores": {sentiment_map[i]: distilbert_scores[i] for i in range(3)}
    }
    # TF-IDF + Logistic Regression
    tfidf_pred = models["tfidf_lr"].predict([text])[0]
    tfidf_probs = models["tfidf_lr"].predict_proba([text])[0]
    results["TF-IDF + LR"] = {
        "prediction": tfidf_pred,
        "confidence": max(tfidf_probs),
        "all_scores": dict(zip(models["tfidf_lr"].classes_, tfidf_probs))
    }

    # BOW + Naive Bayes
    bow_pred = models["bow_nb"].predict([text])[0]
    bow_probs = models["bow_nb"].predict_proba([text])[0]
    results["BOW + NB"] = {
        "prediction": bow_pred,
        "confidence": max(bow_probs),
        "all_scores": dict(zip(models["bow_nb"].classes_, bow_probs))
    }

    return results, cleaned_text


# Main app
def main():
    st.set_page_config(page_title="Car Review Sentiment Analysis", page_icon="🚗", layout="wide")

    st.title("🚗 Car Review Sentiment Analysis")

    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
        df_sample = load_sample_data()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    pages = ["Sentiment Analyzer", "Model Comparison", "Dataset Insights", "About"]
    choice = st.sidebar.radio("Go to", pages)

    if choice == "Sentiment Analyzer":
        st.header("Analyze Car Review Sentiment")

        # Input options
        input_option = st.radio("Choose input type:", ["Enter your own review", "Try example reviews"])

        if input_option == "Enter your own review":
            review_text = st.text_area("Enter a car review:",
                                       height=150,
                                       placeholder="This car has excellent handling and the interior is premium quality...")
        else:
            example_reviews = [
                "This car is absolutely amazing! Great mileage, comfortable seats, and powerful engine.",
                "The car is okay. Nothing special but gets the job done.",
                "Worst car ever! Broke down three times in the first month. Avoid at all costs!",
                "Not bad for the price, but I expected more from this brand. Average performance.",
                "The interior is comfortable, but the engine is noisy and the fuel economy is terrible."
            ]
            review_text = st.selectbox("Choose an example review:", example_reviews)

        if st.button("Analyze Sentiment") and review_text:
            with st.spinner("Analyzing..."):
                results, cleaned_text = analyze_sentiment(review_text, models)

                # Show model predictions
                col1, col2, col3 = st.columns(3)

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.subheader("RoBERTa")
                    color = "green" if results["RoBERTa"]["prediction"] == "Positive" else "red" if results["RoBERTa"][
                                                                                                        "prediction"] == "Negative" else "gray"
                    st.markdown(f"<h3 style='color: {color};'>{results['RoBERTa']['prediction']}</h3>",
                                unsafe_allow_html=True)
                    st.progress(results["RoBERTa"]["confidence"])
                    st.text(f"Confidence: {results['RoBERTa']['confidence']:.2f}")

                with col2:
                    st.subheader("BERT")
                    color = "green" if results["BERT"]["prediction"] == "Positive" else "red" if results["BERT"][
                                                                                                     "prediction"] == "Negative" else "gray"
                    st.markdown(f"<h3 style='color: {color};'>{results['BERT']['prediction']}</h3>",
                                unsafe_allow_html=True)
                    st.progress(results["BERT"]["confidence"])
                    st.text(f"Confidence: {results['BERT']['confidence']:.2f}")

                with col3:
                    st.subheader("DistilBERT")
                    color = "green" if results["DistilBERT"]["prediction"] == "Positive" else "red" if \
                    results["DistilBERT"]["prediction"] == "Negative" else "gray"
                    st.markdown(f"<h3 style='color: {color};'>{results['DistilBERT']['prediction']}</h3>",
                                unsafe_allow_html=True)
                    st.progress(results["DistilBERT"]["confidence"])
                    st.text(f"Confidence: {results['DistilBERT']['confidence']:.2f}")

                with col4:
                    st.subheader("TF-IDF + LR")
                    color = "green" if results["TF-IDF + LR"]["prediction"] == "positive" else "red" if \
                    results["TF-IDF + LR"]["prediction"] == "negative" else "gray"
                    st.markdown(f"<h3 style='color: {color};'>{results['TF-IDF + LR']['prediction']}</h3>",
                                unsafe_allow_html=True)
                    st.progress(results["TF-IDF + LR"]["confidence"])
                    st.text(f"Confidence: {results['TF-IDF + LR']['confidence']:.2f}")

                with col5:
                    st.subheader("BOW + NB")
                    color = "green" if results["BOW + NB"]["prediction"] == "positive" else "red" if \
                    results["BOW + NB"]["prediction"] == "negative" else "gray"
                    st.markdown(f"<h3 style='color: {color};'>{results['BOW + NB']['prediction']}</h3>",
                                unsafe_allow_html=True)
                    st.progress(results["BOW + NB"]["confidence"])
                    st.text(f"Confidence: {results['BOW + NB']['confidence']:.2f}")

                # Show detailed scores
                st.subheader("Sentiment Score Breakdown")
                score_data = []
                for model, data in results.items():
                    for sentiment, score in data["all_scores"].items():
                        score_data.append({"Model": model, "Sentiment": sentiment, "Score": score})

                score_df = pd.DataFrame(score_data)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Model", y="Score", hue="Sentiment", data=score_df, ax=ax)
                plt.title("Sentiment Scores by Model")
                st.pyplot(fig)

                # Text processing details
                with st.expander("See text preprocessing details"):
                    st.subheader("Original Text")
                    st.write(review_text)
                    st.subheader("Cleaned Text")
                    st.write(cleaned_text)

    elif choice == "Model Comparison":
        st.header("Model Performance Comparison")

        # Load performance metrics
        # metrics = {
        #     "BOW + NB": {"Accuracy": 0.82, "F1": 0.81, "Training Time": 2.5},
        #     "TF-IDF + NB": {"Accuracy": 0.83, "F1": 0.82, "Training Time": 3.2},
        #     "BOW + LR": {"Accuracy": 0.84, "F1": 0.83, "Training Time": 5.1},
        #     "TF-IDF + LR": {"Accuracy": 0.86, "F1": 0.85, "Training Time": 6.3},
        #     "GloVe + LR": {"Accuracy": 0.87, "F1": 0.86, "Training Time": 1.2},
        #     "GloVe + NB": {"Accuracy": 0.85, "F1": 0.84, "Training Time": 0.5},
        #     "RoBERTa": {"Accuracy": 0.91, "F1": 0.90, "Training Time": 280}
        # }
        #
        # metrics_df = pd.DataFrame(metrics).T.reset_index()
        # metrics_df = pd.melt(metrics_df, id_vars=["index"], var_name="Metric", value_name="Value")
        # metrics_df.rename(columns={"index": "Model"}, inplace=True)

        comparison_df = models["comparison_df"]
        metrics_df = comparison_df
        metrics_df = pd.melt(metrics_df, id_vars=["index"], var_name="Metric", value_name="Value")
        metrics_df.rename(columns={"index": "Model"}, inplace=True)

        metric_to_show = st.radio("Choose metric to display:", ["Accuracy", "F1", "Training Time"])

        filtered_df = metrics_df[metrics_df["Metric"] == metric_to_show]

        fig, ax = plt.subplots(figsize=(12, 6))
        barplot = sns.barplot(x="Model", y="Value", data=filtered_df, ax=ax)
        plt.title(f"Model Comparison - {metric_to_show}")
        plt.xticks(rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(barplot.patches):
            barplot.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 0.01,
                f"{filtered_df['Value'].iloc[i]:.2f}",
                ha="center", va="bottom"
            )

        st.pyplot(fig)

        # Confusion matrices
        st.subheader("Sample Confusion Matrices")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### TF-IDF + LR")
            # cm_tfidf = np.array([[320, 45, 23], [52, 495, 88], [18, 76, 683]])
            cm_tfidf = models["confusion_matrices"]["TF-IDF + LR"]

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_tfidf, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Neutral', 'Positive'],
                        yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

        with col2:
            st.markdown("#### RoBERTa")
            # cm_roberta = np.array([[345, 32, 11], [38, 525, 72], [12, 59, 706]])
            cm_roberta = models["confusion_matrices"]["RoBERTa"]
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm_roberta, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Negative', 'Neutral', 'Positive'],
                        yticklabels=['Negative', 'Neutral', 'Positive'], ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

    elif choice == "Dataset Insights":
        st.header("Car Review Dataset Insights")

        # Review count by sentiment
        st.subheader("Sentiment Distribution in Dataset")
        sentiment_counts = pd.DataFrame({
            "Sentiment": ["Negative", "Neutral", "Positive"],
            "Count": [8523, 14621, 19144]
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Sentiment", y="Count", data=sentiment_counts, palette=["red", "gray", "green"], ax=ax)
        for i, bar in enumerate(ax.patches):
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + 100,
                f"{sentiment_counts['Count'].iloc[i]:,}",
                ha="center", va="bottom"
            )
        st.pyplot(fig)

        # Reviews by brand
        st.subheader("Top Car Brands by Review Count")
        brand_tab, model_tab, year_tab = st.tabs(["Brands", "Models", "Years"])

        with brand_tab:
            brand_counts = pd.DataFrame({
                "Brand": ["Toyota", "Honda", "Ford", "BMW", "Mercedes", "Audi", "Chevrolet", "Nissan"],
                "Count": [5420, 4983, 4326, 3978, 3652, 3210, 2896, 2765]
            })

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x="Brand", y="Count", data=brand_counts, ax=ax)
            plt.xticks(rotation=45)
            for i, bar in enumerate(ax.patches):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 50,
                    f"{brand_counts['Count'].iloc[i]:,}",
                    ha="center", va="bottom"
                )
            st.pyplot(fig)

        with model_tab:
            model_counts = pd.DataFrame({
                "Model": ["Civic", "Camry", "Accord", "F-150", "Corolla", "3 Series", "Mustang", "Altima"],
                "Count": [2105, 1983, 1826, 1738, 1650, 1523, 1487, 1320]
            })

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x="Model", y="Count", data=model_counts, ax=ax)
            plt.xticks(rotation=45)
            for i, bar in enumerate(ax.patches):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 50,
                    f"{model_counts['Count'].iloc[i]:,}",
                    ha="center", va="bottom"
                )
            st.pyplot(fig)

        with year_tab:
            year_counts = pd.DataFrame({
                "Year": ["2007", "2008", "2009", "2010"],
                "Count": [9823, 13652, 12485, 6828]
            })

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x="Year", y="Count", data=year_counts, ax=ax)
            for i, bar in enumerate(ax.patches):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 50,
                    f"{year_counts['Count'].iloc[i]:,}",
                    ha="center", va="bottom"
                )
            st.pyplot(fig)

        # Important features for sentiment
        st.subheader("Important Words for Each Sentiment")
        col1, col2, col3 = st.columns(3)

        # Word clouds
        with col1:
            st.markdown("#### Positive Reviews")
            positive_words = "excellent great amazing fantastic wonderful comfortable reliable smooth perfect powerful impressive handling quiet exceptional stylish elegant spacious quality premium luxury performance efficient economical satisfied solid fun enjoyable responsive"
            wordcloud = WordCloud(width=400, height=400, background_color='white', colormap='Greens').generate(
                positive_words)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown("#### Neutral Reviews")
            neutral_words = "okay decent average alright fair reasonable acceptable fine standard normal good bad adequate mediocre typical common basic regular expected usual ordinary middle sufficient consistent moderate"
            wordcloud = WordCloud(width=400, height=400, background_color='white', colormap='Blues').generate(
                neutral_words)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        with col3:
            st.markdown("#### Negative Reviews")
            negative_words = "terrible horrible awful poor disappointing bad worse worst problem issue unreliable noisy uncomfortable rough expensive overpriced breakdown repair malfunction costly inefficient loud bumpy cheap plastic flimsy ineffective failure shortage"
            wordcloud = WordCloud(width=400, height=400, background_color='white', colormap='Reds').generate(
                negative_words)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    else:  # About page
        st.header("About This Project")

        st.markdown("""
        ### Car Review Sentiment Analysis

        This project analyzes car reviews to determine sentiment (positive, negative, or neutral) using various natural language processing techniques.

        #### Models Implemented:
        - Traditional machine learning approaches:
          - Bag of Words + Naive Bayes
          - TF-IDF + Naive Bayes
          - Bag of Words + Logistic Regression
          - TF-IDF + Logistic Regression
          - GloVe embeddings + Logistic Regression
          - GloVe embeddings + Gaussian Naive Bayes
        - Deep learning:
          - RoBERTa transformer model
          - BERT transformer model
          - DistilBERT transformer model

        #### Data:
        The models were trained on a dataset containing over 40,000 car reviews covering multiple brands and models from 2007-2009.

        #### Applications:
        - Automate sentiment analysis of customer feedback
        - Identify strengths and weaknesses in specific car models
        - Track consumer sentiment trends over time
        - Compare consumer perception across different car brands

        #### Technologies Used:
        - Python for data analysis and modeling
        - NLTK and spaCy for natural language processing
        - scikit-learn for classical machine learning models
        - Transformers library for RoBERTa implementation
        - Streamlit for this interactive web application
        """)


if __name__ == "__main__":
    main()