import tiktoken

from file_readers import TweetReader


def get_cost_of_encoding_tweets(file_reader, file_path):
    # file_reader is a function that reads a csv file and returns a list of tweets
    num_tweets = 0
    num_tokens = 0
    enc = tiktoken.get_encoding("cl100k_base")
    tweets = file_reader(file_path)
    print(tweets[0])
    for tweet in tweets:
        tweet = tweet['tweet']
        encoding = enc.encode(tweet)
        num_tokens += len(encoding)
        num_tweets += 1

    print(f"Total number of tweets: {num_tweets}\nTotal number of tokens: {num_tokens}")
    print(f"Average number of tokens per tweet: {num_tokens / num_tweets}")

    # calculate cost of encoding all tweets
    # text-embedding-3-small costs $0.00002 per 1k tokens
    # text-embedding-3-large costs $0.00013 per 1k tokens
    print(f"Cost of encoding all tweets with text-embedding-3-small: ${0.00002 * (num_tokens / 1000)}")
    print(f"Cost of encoding all tweets with text-embedding-3-large: ${0.00013 * (num_tokens / 1000)}")



if __name__ == "__main__":
    get_cost_of_encoding_tweets(TweetReader.read_english_training_tweets_csv, "en_training.csv")

