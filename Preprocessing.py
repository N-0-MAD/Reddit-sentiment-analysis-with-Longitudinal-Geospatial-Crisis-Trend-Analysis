import pandas as pd
import re
import emoji
from nltk.corpus import stopwords

# Define the STOPWORDS (you may use your own list if needed)
STOPWORDS = set(stopwords.words("english"))

# Define the function to clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = emoji.demojize(text)  # Replace emojis with text
    text = re.sub(r"http\S+|www\S+", "<URL>", text)  # Replace URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabet characters
    text = " ".join(word for word in text.split() if word not in STOPWORDS)  # Remove stopwords
    return text  
   

# Sample preprocessing script for cleaning
def preprocess_data(reddit_df):

    reddit_df["Cleaned_Title"] = reddit_df["Title"].apply(clean_text)
    reddit_df["Cleaned_Content"] = reddit_df["Content"].apply(clean_text)
    
    reddit_df = reddit_df.drop(columns=['Content', 'Title'])
    
  
    reddit_df = reddit_df.drop_duplicates(subset=["Cleaned_Title", "Cleaned_Content"], keep="first")
    
    # Drop rows with missing values in the cleaned title or content
    reddit_df = reddit_df.dropna(subset=["Cleaned_Title", "Cleaned_Content"])
    
    # Combine the cleaned title and content into a new column
    reddit_df["Combined_Text"] = reddit_df["Cleaned_Title"] + " " + reddit_df["Cleaned_Content"]
    
    return reddit_df


