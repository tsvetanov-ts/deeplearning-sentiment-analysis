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
import os
import plotly.express as px
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
    detailed_metrics = pd.read_csv("detailed_metrics.csv")
    sentiment_counts = pd.read_csv("sentiment_distribution.csv")

    # Load traditional ML models
    # tfidf_lr_pipeline = pickle.load(open("tfidf_lr_model.pkl", "rb"))
    bow_nb_pipeline = pickle.load(open("bow_nb_model.pkl", "rb"))

    lr_model_path = "tfidf_lr_model.pkl"  # Path to your saved model
    if os.path.exists(lr_model_path):
        lr_model_tfidf = pickle.load(open(lr_model_path, "rb"))
        #st.success("Model loaded successfully!")
    else:
        st.error(f"Model file {lr_model_path} not found. Please train the model first.")
        lr_model_tfidf = None


    bow_nb_path = "bow_nb_model.pkl"  # Path to your saved model
    if os.path.exists(bow_nb_path):
        bow_nb_model = pickle.load(open(bow_nb_path, "rb"))
        #st.success("Model loaded successfully!")
    else:
        st.error(f"Model file {bow_nb_path} not found. Please train the model first.")
        bow_nb_model = None

    with open("confusion_matrices.json", "r") as f:
        confusion_matrices = json.load(f)
        for model in confusion_matrices:
            confusion_matrices[model] = np.array(confusion_matrices[model])

    with open("feature_importance.json", "r") as f:
        feature_importance = json.load(f)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    return {
        "tokenizer": tokenizer,
        "roberta_model": roberta_model,
        "bert_tokenizer": bert_tokenizer,
        "bert_model": bert_model,
        "distilbert_tokenizer": distilbert_tokenizer,
        "distilbert_model": distilbert_model,
        "tfidf_lr": lr_model_tfidf,
        "bow_nb": bow_nb_model,
        "lemmatizer": lemmatizer,
        "stop_words": stop_words,
        # "comparison_df": comparison_df,
        "detailed_metrics": detailed_metrics,
        "sentiment_counts": sentiment_counts,
        "confusion_matrices": confusion_matrices,
        "feature_importance": feature_importance

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

    #BOW + Naive Bayes
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

    # elif choice == "Model Comparison":

    elif choice == "Model Comparison":
        st.title("Model Comparison")

        # Load model metrics from CSV
        try:
            model_metrics = pd.read_csv('model_metrics.csv')
            detailed_metrics = pd.read_csv('detailed_metrics.csv')

            # Load confusion matrices from JSON
            with open('confusion_matrices.json', 'r') as f:
                confusion_matrices = json.load(f)

            # Load feature importance if available
            feature_importance = None
            try:
                with open('feature_importance.json', 'r') as f:
                    feature_importance = json.load(f)
            except:
                pass

            # Load sentiment distribution
            try:
                sentiment_distribution = pd.read_csv('sentiment_distribution.csv')
            except:
                sentiment_distribution = None

            # Display overall metrics
            st.header("Model Performance Comparison")

            # Format the dataframe for display
            display_metrics = model_metrics.copy()
            display_metrics['Accuracy'] = display_metrics['Accuracy'].map(lambda x: f"{x:.4f}")
            display_metrics['F1 Score (weighted)'] = display_metrics['F1 Score (weighted)'].map(lambda x: f"{x:.4f}")

            st.dataframe(display_metrics, use_container_width=True)

            # Visualizations
            st.subheader("Performance Metrics Visualization")

            col1, col2 = st.columns(2)

            with col1:
                # Convert to numeric for plotting
                model_metrics['Accuracy'] = pd.to_numeric(model_metrics['Accuracy'])

                fig = px.bar(
                    model_metrics,
                    x='Model',
                    y='Accuracy',
                    title="Model Accuracy Comparison",
                    color='Model',
                    text_auto='.4f'
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                model_metrics['F1 Score (weighted)'] = pd.to_numeric(model_metrics['F1 Score (weighted)'])

                fig = px.bar(
                    model_metrics,
                    x='Model',
                    y='F1 Score (weighted)',
                    title="Model F1 Score Comparison",
                    color='Model',
                    text_auto='.4f'
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics by class
            st.header("Performance by Sentiment Class")

            selected_metric = st.selectbox(
                "Choose metric to visualize:",
                ["Precision", "Recall", "F1-Score"]
            )

            fig = px.bar(
                detailed_metrics,
                x="Model",
                y=selected_metric,
                color="Class",
                barmode="group",
                title=f"{selected_metric} by Model and Sentiment Class",
                text_auto='.3f'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Confusion matrices
            st.header("Confusion Matrices")

            # Select model for confusion matrix
            model_names = list(confusion_matrices.keys())
            selected_model = st.selectbox("Select model for confusion matrix:", model_names)

            if selected_model in confusion_matrices:
                cm = np.array(confusion_matrices[selected_model])

                # Plot confusion matrix
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=['Negative', 'Neutral', 'Positive'],
                    y=['Negative', 'Neutral', 'Positive'],
                    title=f"Confusion Matrix: {selected_model}"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Feature importance visualization if available
            if feature_importance:
                st.header("Feature Importance")

                importance_type = st.radio(
                    "Feature importance type:",
                    ["TF-IDF Logistic Regression", "Word2Vec Similarities"]
                )

                if importance_type == "TF-IDF Logistic Regression" and 'tfidf_logistic_regression' in feature_importance:
                    sentiment = st.selectbox(
                        "Select sentiment class:",
                        ["negative", "neutral", "positive"]
                    )

                    if sentiment in feature_importance['tfidf_logistic_regression']:
                        features = feature_importance['tfidf_logistic_regression'][sentiment]['positive']

                        # Convert to dataframe for visualization
                        feat_df = pd.DataFrame(features)
                        feat_df = feat_df.sort_values('importance', ascending=False).head(15)

                        fig = px.bar(
                            feat_df,
                            x='importance',
                            y='feature',
                            title=f"Top 15 Important Features for {sentiment.capitalize()} Sentiment",
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                elif importance_type == "Word2Vec Similarities" and 'word2vec_similarities' in feature_importance:
                    term = st.selectbox(
                        "Select term:",
                        list(feature_importance['word2vec_similarities'].keys())
                    )

                    if term in feature_importance['word2vec_similarities']:
                        sim_df = pd.DataFrame(feature_importance['word2vec_similarities'][term])

                        fig = px.bar(
                            sim_df,
                            x='similarity',
                            y='word',
                            title=f"Words Most Similar to '{term}'",
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # Sentiment distribution if available
            if sentiment_distribution is not None:
                st.header("Sentiment Distribution in Dataset")

                fig = px.pie(
                    sentiment_distribution,
                    values='Count',
                    names='Sentiment',
                    title="Distribution of Sentiments in the Dataset",
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error loading model comparison data: {str(e)}")
            st.info(
                "Please make sure model_metrics.csv, detailed_metrics.csv, and confusion_matrices.json files exist in the application directory.")


    elif choice == "Dataset Insights":
        st.title("Dataset Insights")

        # Load dataset if not already loaded
        if 'df' not in st.session_state:
            st.session_state.df = pd.read_csv('Out_182.csv')  # Replace with actual path
            # Convert date column to datetime if needed
            if 'date' in st.session_state.df.columns:
                st.session_state.df['date'] = pd.to_datetime(st.session_state.df['date'])

        df = st.session_state.df

        # Display basic dataset statistics
        st.header("Dataset Overview")
        st.write(f"Total reviews: {df.shape[0]:,}")
        st.write(f"Time span: {df['date'].min().date()} to {df['date'].max().date()}")
        st.write(f"Unique car brands: {df['brand'].nunique()}")
        st.write(f"Unique car models: {df['model'].nunique()}")

        # Display tabs for different insights
        tabs = st.tabs(["Reviews Over Time", "Brands & Models", "Sentiment Analysis", "Favorites"])

        # Reviews Over Time tab
        with tabs[0]:
            st.subheader("Reviews Distribution Over Time")
            reviews_per_day = df.groupby(df['date'].dt.date).size().reset_index(name='reviews')

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(reviews_per_day['reviews'], kde=True, ax=ax)
            ax.set_title('Distribution of Daily Review Counts')
            ax.set_xlabel('Number of Reviews per Day')
            st.pyplot(fig)

            # Top days with most reviews
            st.subheader("Top Days with Most Reviews")
            top_days = reviews_per_day.sort_values('reviews', ascending=False).head(10)
            st.dataframe(top_days)

        # Brands & Models tab
        with tabs[1]:
            st.subheader("Top Brands by Review Count")
            brand_counts = df['brand'].value_counts().head(10)

            fig, ax = plt.subplots(figsize=(10, 6))
            brand_counts.plot.bar(ax=ax)
            ax.set_title('Top 10 Most Reviewed Car Brands')
            ax.set_ylabel('Number of Reviews')
            st.pyplot(fig)

            # Display top models for selected brand
            selected_brand = st.selectbox("Select a brand to see top models:",
                                          df['brand'].value_counts().head(20).index.tolist())

            if selected_brand:
                brand_df = df[df['brand'] == selected_brand]
                model_counts = brand_df['model'].value_counts().head(10)

                fig, ax = plt.subplots(figsize=(10, 6))
                model_counts.plot.bar(ax=ax)
                ax.set_title(f'Top Models for {selected_brand}')
                ax.set_ylabel('Number of Reviews')
                st.pyplot(fig)

        # Sentiment Analysis tab
        with tabs[2]:
            st.subheader("Sentiment Distribution")

            if 'vader_compound' not in df.columns:
                st.warning("Sentiment scores not found in dataset. Run sentiment analysis first.")
            else:
                # Create sentiment label if not already present
                if 'sentiment_label' not in df.columns:
                    df['sentiment_label'] = pd.cut(
                        df['vader_compound'],
                        bins=[-1, -0.1, 0.1, 1],
                        labels=['negative', 'neutral', 'positive']
                    )

                sentiment_counts = df['sentiment_label'].value_counts()

                fig, ax = plt.subplots(figsize=(10, 6))
                sentiment_counts.plot.pie(autopct='%1.1f%%', ax=ax)
                ax.set_title('Review Sentiment Distribution')
                st.pyplot(fig)

                # Display sentiment over time
                st.subheader("Sentiment Trends Over Time")

                sentiment_by_date = df.groupby([df['date'].dt.date, 'sentiment_label']).size().unstack().fillna(0)
                sentiment_by_date = sentiment_by_date.rolling(window=7).mean()  # 7-day moving average

                fig, ax = plt.subplots(figsize=(12, 6))
                sentiment_by_date.plot(ax=ax)
                ax.set_title('7-Day Moving Average of Sentiment Counts')
                ax.set_xlabel('Date')
                ax.set_ylabel('Number of Reviews')
                st.pyplot(fig)

        # Favorites tab
        with tabs[3]:
            st.subheader("Most Common Favorites")

            favorite_counts = df['favorite'].value_counts().reset_index()
            favorite_counts.columns = ['Favorite', 'Count']

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(data=favorite_counts.head(15), x='Favorite', y='Count', ax=ax)
            ax.set_title('Most Common Favorites', fontsize=14)
            ax.set_xlabel('Favorite', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

            # Top favorites table
            st.dataframe(favorite_counts.head(20))

    else:  # About page
        st.header("About This Project")
        st.write("""
        ## Car Reviews Sentiment Analysis

        This application demonstrates advanced natural language processing and sentiment analysis techniques applied to car reviews. 
        The project analyzes consumer opinions to extract meaningful insights about vehicle preferences and satisfaction.

        ### Project Overview
        - **Data Source**: Collection of car reviews with ratings, comments, and metadata
        - **Analysis Goal**: Classify sentiment in car reviews as positive, neutral, or negative
        - **Features**: Text analysis, sentiment classification, and visualization

        ### Methodology
        The project implements and compares multiple text classification approaches:
        - **Traditional ML Models**:
          - Bag-of-Words with Naive Bayes
          - TF-IDF with Naive Bayes
          - TF-IDF with Logistic Regression
        - **Deep Learning Models**:
          - BERT
          - DistilBERT
          - RoBERTa

        ### Results
        - Transformer models (BERT, RoBERTa) achieved the highest accuracy for sentiment classification
        - Important features related to car performance, reliability, and comfort were identified
        - Text embeddings revealed patterns in how consumers describe their vehicles

        ### Insights
        This analysis helps understand consumer preferences and sentiment around different vehicle brands, models, and features.
        """)

        st.write("---")
        st.write("Created by: Tsvetanov, Tsvetan Tsvetanov")
        st.write("Sentiment Analysis Project, 2025")
        st.write("Introduction to Deep Learning, Faculty of Mathematics and Informatics, Sofia University, Bulgaria")


if __name__ == "__main__":
    main()