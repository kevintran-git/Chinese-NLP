import csv

class TweetReader:
    @staticmethod
    def read_training_tweets_csv(file_path):
        tweets = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 3: # Make sure row has exactly three elements
                        tweet = row[0]
                        rating_pos = row[1]
                        rating_neg = row[2]

                        # make sure all elements are non null
                        if tweet != '' and rating_pos != '' and rating_neg != '':
                            tweets.append({'tweet': tweet, 'rating_pos': rating_pos, 'rating_neg': rating_neg})
        except Exception as e:
            print(f"Failed to read file: {e}")
        return tweets
    
    @staticmethod
    def read_english_training_tweets_csv(file_path):
        tweets = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.reader(file)
                for row in reader:
                    if len(row) == 3: # Make sure row has exactly three elements
                        tweet = row[0]
                        rating_pos = row[1]
                        rating_neg = row[2]

                        # make sure all elements are non null
                        if tweet != '' and rating_pos != '' and rating_neg != '':
                            # convert negative rating to positive
                            rating_neg = abs(int(rating_neg))
                            tweets.append({'tweet': tweet, 'rating_pos': rating_pos, 'rating_neg': rating_neg})
        except Exception as e:
            print(f"Failed to read file: {e}")
        return tweets

    @staticmethod
    def read_test_tweets_csv(file_path):
        tweets = []
        try:
            with open(file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)  # Using DictReader to easily access columns by names
                for row in reader:
                    # Extracting required fields from each row
                    author_id = row['author id']
                    created_at = row['created_at']
                    geo = row['geo']
                    tweet_id = row['id']
                    lang = row['lang']
                    like_count = row['like_count']
                    quote_count = row['quote_count']
                    reply_count = row['reply_count']
                    retweet_count = row['retweet_count']
                    tweet = row['tweet']

                    # Assuming you want to keep all tweets regardless of certain conditions
                    tweets.append({
                        'author_id': author_id,
                        'created_at': created_at,
                        'geo': geo,
                        'tweet_id': tweet_id,
                        'lang': lang,
                        'like_count': like_count,
                        'quote_count': quote_count,
                        'reply_count': reply_count,
                        'retweet_count': retweet_count,
                        'tweet': tweet
                    })
        except Exception as e:
            print(f"Failed to read file: {e}")
        return tweets

