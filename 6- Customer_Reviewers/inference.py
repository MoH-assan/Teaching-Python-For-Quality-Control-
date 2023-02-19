"""
Introduction: 
This code use a pre-trained machine learning model to automate the quality control of customer satisfaction.
This code can be integrated, to statistical process control (SPC) package  to do the np chart 
to identify any changes in product quality and take corrective action if necessary.

Objective:
 Create an np chart for quality control purposes.

Input:
-CSV file with the reviews text.

Steps:
-Install required packages and load the tokenizer and model with the Transformers library.
-Define a predict_sentiment function that returns a predicted sentiment label.
-Test predict_sentiment function with example review texts, have fun !
-Load the real Amazon reviews dataset and apply predict_sentiment function to each review.
-Store predicted sentiments in a new DataFrame and export to CSV file.

Output:
-CSV file with the predicted sentiments for each review.

Further steps:
-Write the code to create the np chart.
-Use ground truth labels to evaluate the model performance. Your SPC tool is only as good as your NLP model.
"""

import subprocess
import sys
# %%
# Check if the required packages are installed, and install them if necessary
def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

install("torch")
install("transformers")
install("pandas")

# Import the required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# Load the pre-trained tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english", num_labels=2)

# Define the predict_sentiment function that takes a review text as input and returns the predicted sentiment
def predict_sentiment(review_text):
    # Encode the review text using the pre-trained tokenizer
    input_ids = tokenizer.encode(review_text, padding=True, truncation=True, return_tensors="pt")
    # Feed the encoded review text to the pre-trained model to get the predicted sentiment
    outputs = model(input_ids)
    predicted_class = torch.argmax(outputs.logits).item()
    # Map the predicted sentiment to a string label ("Positive" or "Negative") and return it
    if predicted_class == 0:
        return "Negative"
    else:
        return "Positive"
# %%
# Test the predict_sentiment function with some example review texts, Have fun and try to confuse the model, share with us if you can, maybe you get a bonous!
review1 = "my father can not stopping using it"; review2 = "my kid injured himself with this product" 
review3="I take 1 star off, because I had problems to syncing it with my W-FI" # What do you think about this review?, see what the model thinks about it.

print(f"Review: {review1} | Sentiment: {predict_sentiment(review1)}")
print(f"Review: {review2} | Sentiment: {predict_sentiment(review2)}")
# The model shows a good performance, howerer this is not how machine learning expert test the model performance, but it is enough for our purpose here.
# %%
# Let's look at some real amazon reviews.
reviews=pd.read_csv("reviews_only.csv") # Load the cleaned data set. 
#The orginal data was downloaded from here: https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products
print(reviews.head()) #Each raw is a sample of size 10 (i.e. 10 reviews per day, ) 
print(reviews.shape) #In total we have 5000 samples (i.e. 500 days)
PRINT_FLAG,PRINT_FREQ=True,100 # Some parameters to control the printing, don't worry about them
# %% Instead of asking a human to read all these reviews and classify the sentiment  (i.e., decide if it is postive or negative), let's ask our model to do it for us.
# This takes a few minute to run (5 mins on my not so powerful laptop), remember we have 500 days each day has 10 reviews
# To keep you entertained, I will print the predicted sentiment for some days, you can check them while waiting.

# Now let the show begin
# We will iterate over all days and all reviews at each day and predict the sentiment for each review
list_pred_sentiment_all_days=[] # to collect the model predictions for all days
for day_number,each_day_all_reviews in reviews.iterrows(): # iterate over all rows (i.e.,days)
    list_pred_sentiment_one_day=[]  # to collect the model predictions for this particular day
    for rewiew_number_at_that_day, single_review_at_that_day in enumerate(each_day_all_reviews): # iterate over all reviews at that single day
        predicted_sentiment=predict_sentiment(single_review_at_that_day)
        list_pred_sentiment_one_day.append(predicted_sentiment) 
        if (PRINT_FLAG) and (day_number%PRINT_FREQ==0): # Just reduce the number of prints by prints once every 100 days
            print(f"day {day_number}")
            print(f"Review : {single_review_at_that_day}")
            print(f"Predicted Sentiment: {predicted_sentiment}")
            if rewiew_number_at_that_day==reviews.shape[1]:
                print("=================Day Ended========================")
    list_pred_sentiment_all_days.append(list_pred_sentiment_one_day)           
# %%
## % [markdown]
# ## 4. Save the predicted sentiment for each review
predicted_sentiments=pd.DataFrame(list_pred_sentiment_all_days)
predicted_sentiments.columns=reviews.columns
print(predicted_sentiments.head())
print(predicted_sentiments.shape)   
print(predicted_sentiments.tail)

predicted_sentiments.to_csv("predicted_sentiments.csv",index=False)

# %%
# Now that we have the predicted sentiment for each review, 
# We can forget about the ML model and go back to the quality control tool. 
# For example to do the np chart, you can forget about everything and just use the predicted_sentiments.csv file.