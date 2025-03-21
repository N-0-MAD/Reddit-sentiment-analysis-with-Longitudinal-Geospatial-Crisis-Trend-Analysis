# **Reddit Sentiment Analysis with Longitudinal & Geospatial Crisis Trend Analysis**  

## **Overview**  
A significant amount of research has been conducted on sentiment analysis of social media posts, particularly on platforms like Reddit and Twitter. Many state-of-the-art methods leverage deep learning techniques for enhanced sentiment classification.This project aims to analyze Reddit posts to classify sentiments and assess crisis-related discussions.  
In this project, I have fine-tuned a **DistilBERT** model on a dataset of 1,825 Reddit posts, collected using the Reddit API. The sentiment analysis is performed on the combined title and content of each post to gain deeper insights.
For sentiment classification, I have used **TextBlob** to categorize posts as positive, negative, or neutral. Additionally, the fine-tuned DistilBERT model is employed to classify the risk level of each post, helping to identify crisis-related discussions.

## **Features**  
- **Sentiment Analysis**: Classifies posts as **Positive, Negative, or Neutral** using TextBlob.  
- **Crisis Risk Classification**: Uses a **fine-tuned DistilBERT model** to categorize posts into different risk levels.  
- **Geolocation Mapping**: Extracts locations from posts and plots them on an **interactive heatmap** using Folium.  

## **Installation**  

### **1. Clone the Repository**  
Open a terminal and run:  
```bash
git clone https://github.com/N-0-MAD/Reddit-sentiment-analysis-with-Longitudinal-Geospatial-Crisis-Trend-Analysis.git
cd Reddit-sentiment-analysis-with-Longitudinal-Geospatial-Crisis-Trend-Analysis
```

### **2. Create and Activate a Virtual Environment**  
```bash
python -m venv venv  
source venv/bin/activate  # For Mac/Linux  
venv\Scripts\activate  # For Windows  
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Download Required NLTK & SpaCy Models**  
Run the provided script to automatically download the necessary models:  

```bash
python setup_nltk_spacy.py
```
This script will ensure that all required **NLTK** and **SpaCy** models are installed and ready for use.
Although I have mentioned all the external dependencies used in this project, issues like numpy version conflict may occur. If this error occurs then try downgrading numpy version  to 1.19.5. 
If any dependency is not installed then you may install it using pip.
## **Usage**  

Your dataset must have the following columns: columns=["ID", "Timestamp", "Title", "Content", "Upvotes", "Comments", "URL"]
If you don't have the data, you can use Reddit API key to get the posts. 
**The code snippet for fetching reddit posts is available in the file: reddit data.ipynb**
Once you get the dataset, do the following procedure preferably in a Jupyter Notebook of same directory:

### **1. Preprocess the Data**  
```Jupyter notebook
import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords
import praw
import time
import torch
import Preprocessing

reddit_df = pd.read_csv("reddit_exp.csv")
reddit_df["Title"] = reddit_df["Title"].astype(str)
reddit_df["Content"] = reddit_df["Content"].astype(str)
reddit_df = Preprocessing.preprocess_data(reddit_df)
```
This script cleans the dataset and prepares it for sentiment analysis.[reddit_df]  

### **2. Classify Sentiments & Risk Levels**  
```Jupyter Notebook
import classify_post
reddit_df_classified= classify_post.process_and_visualize(reddit_df)
reddit_df_classified.to_csv("classified_risk_levels.csv", index=False)
```
This script uses TextBlob and DistilBERT to classify the sentiment and crisis risk level of posts. 
reddit_df_classified is the final dataframe having sentiments and risk levels (0=low risk, 1=moderate concern, 2=high risk)

### **3. Perform Geolocation Analysis & Generate a Heatmap**  
```Jupyter Notebook
from crisis_geolocation import generate_crisis_heatmap
crisis_geolocation.generate_crisis_heatmap(reddit_df_classified)
```
This script extracts location-related information and plots high-risk discussion points on a heatmap (saved as `crisis_heatmap.html`). 
Also, it will give the top 5 locations as well.

## **Results**  
- **Crisis Heatmap**: Highlights key locations where crisis-related discussions are prominent.  
- **Sentiment Classification**: Categorizes Reddit posts based on polarity.  
- **Risk Classification**: Uses deep learning (DistilBERT) to determine the urgency of posts.   

## **Example**  
An example of execution of above steps is given in the file 'Experiment.ipynb'

## **Improvements**
The current model achieved an accuracy of 0.76. We can improve this model further by integrating Bert-CNN and other deep learning approaches. 
Also I am planning to add more data regularly and re-tune the model for better accuracy once I have around 2-3 lakh posts data. I am sure that we can achieve an accuracy > 0.8 by adding more data.
