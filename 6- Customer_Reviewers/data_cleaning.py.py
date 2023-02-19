
import requests
import zipfile
import io
import pandas as pd
# %%
# %%
# Load the dataset into a Pandas DataFrame
df = pd.read_csv("Consumer_Reviews_of_Amazon_Products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")

# Keep only the necessary columns and clean the data
df = df[["reviews.text", "reviews.rating"]]
df = df.dropna()

def label_sentiment(rating):
    # Convert numerical rating to sentiment label
    if rating > 3:
        return "1" # postive
    elif rating == 3:
        return "0" # neutral
    else:
        return "0" # negative, note we are here treating neutral as negative

df["sentiment"] = df["reviews.rating"].apply(label_sentiment)
df = df[["reviews.text", "sentiment"]]
df.to_csv("reviews_sentiment.csv", index=False)
# %%

samples_size=10 # Our choice
number_samples=int(len(df)/samples_size) # neglect the lsat few samples that are less than 1000
column_names_reviews=[f"review_{i}" for i in range(samples_size)]
column_names_setiment=[f"review_{i}_setiment" for i in range(samples_size)]


df=df[:number_samples*samples_size]
df_review_reshaped = pd.DataFrame(df["reviews.text"].values.reshape(-1, samples_size))
df_review_reshaped.columns=column_names_reviews

df_sentiment_reshaped= pd.DataFrame(df["sentiment"].values.reshape(-1, samples_size))
df_sentiment_reshaped.columns=column_names_setiment

df2=pd.concat([df_review_reshaped, df_sentiment_reshaped], axis=1)
df2.index=[f"sample_{i}" for i in range(number_samples)]
df2.to_csv("reviews_sentiments.csv", index=False)

df_review_reshaped.to_csv("reviews_only.csv", index=False)
df_sentiment_reshaped.to_csv("sentiments_only.csv", index=False)
