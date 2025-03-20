import pandas as pd
import re
import emoji
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  
    text = emoji.demojize(text)  
    text = re.sub(r"http\S+|www\S+", "<URL>", text) 
    text = re.sub(r"[^a-zA-Z\s]", "", text) 
    text = " ".join(word for word in text.split() if word not in STOPWORDS) 
    return text  
   

def preprocess_data(reddit_df):

    reddit_df["Cleaned_Title"] = reddit_df["Title"].apply(clean_text)
    reddit_df["Cleaned_Content"] = reddit_df["Content"].apply(clean_text)
    
    reddit_df = reddit_df.drop(columns=['Content', 'Title'])
    
  
    reddit_df = reddit_df.drop_duplicates(subset=["Cleaned_Title", "Cleaned_Content"], keep="first")
    

    reddit_df = reddit_df.dropna(subset=["Cleaned_Title", "Cleaned_Content"])
    
    reddit_df["Combined_Text"] = reddit_df["Cleaned_Title"] + " " + reddit_df["Cleaned_Content"]
    
    return reddit_df


