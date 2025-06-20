# Sentiment Analysis Project
## About
This project is designed to perform sentiment analysis on dataset with car reviews for 2007. 2008 and 2009 models. It utilizes various machine learning models to classify the sentiment of the text as positive, negative, or neutral.
## Data
Dataset used here is taken from [OpinRank Review Dataset](https://archive.ics.uci.edu/dataset/205/opinrank+review+dataset)
More details can be found in `data/OpinRank_Dataset_Description.pdf`
## Technical information

Main project file is `sentiment_analysis.ipynb` notebook. All the messy code is there. It:
* loads data
* preprocesses text
* performs analytics
* performs feature engineering
* draws plots
* trains models
* evaluates models
* saves results

The other important file is `streamlit_gui.py` which is a Streamlit app. 
It visualizes the results of the analysis and allows users to interact with the model.

## Saved data
Each models saves its result after running. ML models  - Logistic Regression and Naive Bayes save pickles (.pkl) files which are small and can be easily loaded and reused. Therefore I committed those to the repo. However neural networks produce large files, so I did not commit them. GitHub has 100M file limit. You can run the notebook to train them and save on your machine. 
Almost everything ran on my macbook. However the RoBERTa was trained on A100 GPU on Google Colab as it would take 6 hours using Apple Metal backend and would potentially crash due to various reasons such as memory overflow.

Deep Learning models used:
* RoBERTa
* BERT
* DistilBERT

Optimization techniques used:
* Optimizer: AdamW
* Optuna for hyperparameter tuning (RoBERTa only)

## Colab notebook

The Jupyter notebook from Google Colab is also available in the repo. It is named `sentiment_analysis_colab.ipynb`. Here's link to Colab: [Sentiment Analysis Colab](https://colab.research.google.com/github/yourusername/sentiment_analysis/blob/main/sentiment_analysis_colab.ipynb)