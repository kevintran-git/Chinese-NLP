import csv
import tqdm
import google.generativeai as genai
from openai import OpenAI
from auth import genai_key, sk_key
import re
from file_readers import TweetReader

def embed_tweets_google(tweets, file_path, is_test=False):
    genai.configure(api_key=sk_key)
    for tweet in tqdm.tqdm(tweets):
        tweet['tweet'] = tweet['tweet'].replace('\n', ' ')
        embedding = genai.embed_content(model='models/embedding-001',
                                        content=tweet['tweet'],
                                        task_type="classification")
        embedding = embedding['embedding']
        print(f"Embedding for '{tweet['tweet']}': {embedding}")
        tweet['embedding'] = embedding
    
    try:
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            for tweet in tweets:
                if is_test:
                    writer.writerow([tweet['tweet'], tweet['embedding']])
                else:
                    writer.writerow([tweet['tweet'], tweet['embedding'], tweet['rating_pos'], tweet['rating_neg']])
    except Exception as e:
        print(f"Failed to write file: {e}")
    return tweets

def embed_tweets_openai(tweets, file_path, batch_size=100):
    client = OpenAI(api_key=sk_key)
    try:
        with open(file_path, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            for batch_start in tqdm.tqdm(range(0, len(tweets), batch_size)):
                batch_end = min(batch_start + batch_size, len(tweets))
                tweet_batch = tweets[batch_start:batch_end]

                # Preprocess tweets in the batch
                processed_tweets = []
                for tweet in tweet_batch:
                    # Remove mentions
                    tweet['tweet'] = re.sub(r'@\w+', '', tweet['tweet']).strip()  
                    # Remove URLs
                    tweet['tweet'] = re.sub(r'http\S+', '', tweet['tweet']).strip()
                    # Remove <br> tags
                    tweet['tweet'] = re.sub(r'<br>', '', tweet['tweet']).strip()
                    # Remove escape characters such as quotes, tabs, and newlines
                    tweet['tweet'] = re.sub(r'\\[\'"\\nrt]', '', tweet['tweet']).strip()
                    # Replace newlines with spaces
                    tweet['tweet'] = tweet['tweet'].replace('\n', ' ')  
                    processed_tweets.append(tweet['tweet'])

                # Request embeddings for the batch
                try:
                    embeddings_response = client.embeddings.create(
                        input=processed_tweets,
                        model="text-embedding-3-large",
                    ).data
                except Exception as e:
                    print(f"Failed to create embeddings: {e}")
                    print(f"Tweet that caused the error: {tweet['tweet']}")
                    continue

                # Process each tweet and its embedding
                for i, tweet in enumerate(tweet_batch):
                    output = embeddings_response[i].embedding
                    #print(f"Embedding for '{tweet['tweet']}': {output[:10]}")
                    tweet['embedding'] = output

                    # Determine if it's a test case based on presence of rating keys
                    if 'rating_pos' in tweet and 'rating_neg' in tweet:
                        writer.writerow([tweet['tweet'], tweet['embedding'], tweet['rating_pos'], tweet['rating_neg']])
                    else:
                        writer.writerow([tweet['tweet'], tweet['embedding']])
                # Flush file after processing each batch
                file.flush()

    except Exception as e:
        print(f"Failed to write file: {e}")
    return tweets


if __name__ == "__main__":
    tweets = TweetReader.read_english_training_tweets_csv("en_training.csv")
    embed_tweets_openai(tweets, 'en_large_embeddings.csv')
    
    
