import csv 
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import matplotlib.pyplot as plt

# -------------------------------------- Data Reading -------------------------------------- #

def read_embeddings_csv(file_path):
    embeddings = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4:  # Make sure row has exactly four elements
                    tweet = row[0] # example: "你也去了吗"
                    embedding = row[1] #  returns "{'embedding': [0.1, 0.2, 0.3, ...]}" as a string

                    # turn embedding into a vector
                    embedding = embedding.replace("[", "")
                    embedding = embedding.replace("]", "")
                    embedding = embedding.split(", ")
                    embedding = [float(i) for i in embedding] # example: [0.1, 0.2, 0.3, ...]

                    # these are integers from 1 to 5
                    rating_pos = int(row[2])
                    rating_neg = int(row[3])

                    # make sure all elements are non null
                    if tweet != '' and embedding != '' and rating_pos != '' and rating_neg != '':
                        embeddings.append({'tweet': tweet, 'embedding': embedding, 'rating_pos': rating_pos, 'rating_neg': rating_neg})
    except Exception as e:
        print(f"Failed to read file: {e}")
    return embeddings

def read_test_tweets_csv(file_path):
    # test tweets are used to evaluate the model, they don't have ratings
    # format: tweet, embedding (vector as string)
    tweets = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2:  # Make sure row has exactly two elements
                    tweet = row[0] # example: "你也去了吗"
                    embedding = row[1] #  returns "{'embedding': [0.1, 0.2, 0.3, ...}" as a string

                    # turn embedding into a vector
                    embedding = embedding.replace("[", "")
                    embedding = embedding.replace("]", "")
                    embedding = embedding.split(", ")
                    embedding = [float(i) for i in embedding] # example: [0.1, 0.2, 0.3, ...]

                    # make sure all elements are non null
                    if tweet != '' and embedding != '':
                        tweets.append({'tweet': tweet, 'embedding': embedding})
    except Exception as e:
        print(f"Failed to read file: {e}")
    return tweets

class TweetDataset(Dataset):
    def __init__(self, tweets):
        self.tweets = tweets

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        embedding = torch.tensor(tweet['embedding'], dtype=torch.float)
        # label_pos will be 0 if the rating is 1, 1 if the rating is 2, 2 if the rating is 3 or higher. 4 is 2, 5 is 2
        def convert_rating(rating):
            if rating == 1:
                return 0
            elif rating == 2:
                return 1
            else:
                return 2
        label_pos = convert_rating(tweet['rating_pos'])
        label_neg = convert_rating(tweet['rating_neg'])
        return embedding, label_pos, label_neg

def initialize_data_loader(tweets):
    tweets_train, tweets_val = train_test_split(tweets, test_size=0.2, random_state=10) # prev: 42
    train_dataset = TweetDataset(tweets_train)
    val_dataset = TweetDataset(tweets_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, len(tweets_train[0]['embedding'])

# -------------------------------------- Model -------------------------------------- #

class TextSentimentClassifier(nn.Module):
    def __init__(self, input_size):
        super(TextSentimentClassifier, self).__init__()
        
        # Shared layers
        self.shared_layer1 = nn.Linear(input_size, 1024)
        self.shared_dropout1 = nn.Dropout(0.6)  # Increased dropout
        self.shared_layer2 = nn.Linear(1024, 512)
        self.shared_dropout2 = nn.Dropout(0.6)  # Increased dropout
        self.batch_norm1 = nn.BatchNorm1d(512)  # Added batch normalization
        
        # Positive sentiment branch
        self.pos_layer1 = nn.Linear(512, 256)
        self.pos_dropout1 = nn.Dropout(0.5)
        self.pos_output = nn.Linear(256, 3)
        
        # Negative sentiment branch
        self.neg_layer1 = nn.Linear(512, 256)
        self.neg_dropout1 = nn.Dropout(0.5)
        self.neg_output = nn.Linear(256, 3)
        
    def forward(self, x):
        # Shared layers
        x = F.relu(self.shared_layer1(x))
        x = self.shared_dropout1(x)
        x = F.relu(self.shared_layer2(x))
        x = self.shared_dropout2(x)
        x = self.batch_norm1(x)
        
        # Positive sentiment branch
        pos = F.relu(self.pos_layer1(x))
        pos = self.pos_dropout1(pos)
        pos = self.pos_output(pos)
        
        # Negative sentiment branch
        neg = F.relu(self.neg_layer1(x))
        neg = self.neg_dropout1(neg)
        neg = self.neg_output(neg)
        
        return pos, neg
    
# -------------------------------------- Training -------------------------------------- #
    
def calculate_accuracy(outputs, labels):
    """
    Calculate the accuracy of the model's predictions.
    
    Args:
        outputs (torch.Tensor): Model's predictions.
        labels (torch.Tensor): Actual labels.
        
    Returns:
        tuple: A tuple containing the number of correct predictions and the total number of samples.
    """
    _, predicted = torch.max(outputs.data, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return correct, total

def epoch_operation(model, loader, criterion, optimizer=None, train=True):
    """
    Perform a single epoch of training or validation.

    This function abstracts the common operations performed during each epoch, whether it's
    a training epoch or a validation/testing epoch. The differences are controlled via the
    `train` flag and the presence of an optimizer.

    Args:
        model (torch.nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the current phase (train or validation).
        criterion (torch.nn): Loss function.
        optimizer (torch.optim.Optimizer, optional): Optimizer for the training phase.
        train (bool, optional): Flag to distinguish between train and validation phase. Defaults to True.

    Returns:
        tuple: A tuple containing the average loss of the epoch, accuracies for positive and negative ratings,
               and lists of actual and predicted labels for both positive and negative ratings.
    """

    if train:
        model.train()
    else:
        model.eval()

    total_acc_pos = total_acc_neg = total_samples = 0
    all_labels_pos, all_predicted_pos, all_labels_neg, all_predicted_neg = [], [], [], []

    for embeddings, label_pos, label_neg in loader:
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            output_pos, output_neg = model(embeddings)
            loss_pos = criterion(output_pos, label_pos)
            loss_neg = criterion(output_neg, label_neg)
            loss = loss_pos + loss_neg
            if train:
                loss.backward()
                optimizer.step()

        correct_pos, total_pos = calculate_accuracy(output_pos, label_pos)
        correct_neg, total_neg = calculate_accuracy(output_neg, label_neg)
        total_acc_pos += correct_pos
        total_acc_neg += correct_neg
        total_samples += total_pos  # Assuming total_pos and total_neg are the same

        all_labels_pos += label_pos.view(-1).tolist()
        all_predicted_pos += output_pos.argmax(1).view(-1).tolist()
        all_labels_neg += label_neg.view(-1).tolist()
        all_predicted_neg += output_neg.argmax(1).view(-1).tolist()

    accuracy_pos = 100 * total_acc_pos / total_samples
    accuracy_neg = 100 * total_acc_neg / total_samples
    return loss.item(), accuracy_pos, accuracy_neg, all_labels_pos, all_predicted_pos, all_labels_neg, all_predicted_neg

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=200, patience=5):
    best_val_acc = 0
    wait = 0  # Counter for patience

    for epoch in range(num_epochs):
        train_loss, train_acc_pos, train_acc_neg, _, _, _, _ = epoch_operation(model, train_loader, criterion, optimizer, train=True)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Acc Pos: {train_acc_pos}%, Train Acc Neg: {train_acc_neg}%')

        _, val_acc_pos, val_acc_neg, all_labels_pos, all_predicted_pos, all_labels_neg, all_predicted_neg = epoch_operation(model, val_loader, criterion, train=False)
        print(f'Epoch {epoch+1}, Val Acc Pos: {val_acc_pos}%, Val Acc Neg: {val_acc_neg}%')

        if val_acc_pos + val_acc_neg > best_val_acc:
            best_val_acc = val_acc_pos + val_acc_neg
            wait = 0
            # model_name = f'model_epoch_{epoch+1}_val_acc_pos_{val_acc_pos}_val_acc_neg_{val_acc_neg}.ckpt'
            # torch.save(model.state_dict(), model_name)
        else:
            wait += 1

        if wait >= patience:
            print("Early stopping!")
            break
    generate_confusion_matrix(all_labels_pos, all_predicted_pos, all_labels_neg, all_predicted_neg)

def generate_confusion_matrix(all_labels_pos, all_predicted_pos, all_labels_neg, all_predicted_neg):
    confusion_matrix_pos = confusion_matrix(all_labels_pos, all_predicted_pos)
    confusion_matrix_neg = confusion_matrix(all_labels_neg, all_predicted_neg)

    print("Confusion Matrix for Positive Ratings:")
    print(confusion_matrix_pos)
    print("\nConfusion Matrix for Negative Ratings:")
    print(confusion_matrix_neg)
    
def classify_and_output_test_data(model_path, test_data_csv, output_file_path):
    # Read test data
    test_tweets = read_test_tweets_csv(test_data_csv)

    # get the length of the embedding
    input_size = len(test_tweets[0]['embedding'])

    # Load the model
    model = TextSentimentClassifier(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Classification and output preparation
    predictions = []
    for tweet in tqdm.tqdm(test_tweets):
        embedding = torch.tensor(tweet['embedding'], dtype=torch.float).unsqueeze(0) # Add batch dimension
        rating_pos, rating_neg = model(embedding)
        predicted_pos = rating_pos.argmax(dim=1).item() + 1  # Adjust for 1-indexed ratings
        predicted_neg = rating_neg.argmax(dim=1).item() + 1
        predictions.append({'tweet': tweet['tweet'], 'rating_pos': predicted_pos, 'rating_neg': predicted_neg})

    # Write predictions to output file
    with open(output_file_path, mode='w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['tweet', 'rating_pos', 'rating_neg'])
        for prediction in predictions:
            writer.writerow([prediction['tweet'], prediction['rating_pos'], prediction['rating_neg']])
    print(f'Predictions written to {output_file_path}')

if __name__ == "__main__":
    tweets = read_embeddings_csv('chinese_twitter_embeddings_openai.csv')
    train_loader, val_loader, input_size = initialize_data_loader(tweets)
    
    model = TextSentimentClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    train_model(model, train_loader, val_loader, criterion, optimizer)

    # model_path = 'model_val_acc_pos_71.03448275862068_val_acc_neg_81.37931034482759.ckpt'
    # # evaluate the model accuracy on training data
    # tweets = read_embeddings_csv('chinese_twitter_embeddings_openai.csv')
    # train_loader, val_loader, input_size = initialize_data_loader(tweets)
    # model = TweetClassifier(input_size)
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # validate(model, val_loader)
    

    #test_data_csv = 'chinese_twitter_embeddings_openai.csv'
    #output_file_path = 'classified_tweets_training.csv'
    #classify_and_output_test_data(model_path, test_data_csv, output_file_path)



