print("Hello Movie Review Sentiment Analysis!")
import nltk                            # This loads the NLTK library for working with text.
from nltk.corpus import movie_reviews # This lets us access the movie reviews dataset.
import random                         # This helps us shuffle data randomly.

# Download the dataset to your computer (only need to do once)
nltk.download('movie_reviews')
# Create a list of tuples: (words_in_review, label)
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()          # 'pos' and 'neg'
    for fileid in movie_reviews.fileids(category)       # each review file
]

# Mix up the order of the reviews randomly
random.shuffle(documents)
print(f"Number of reviews: {len(documents)}")          # Total reviews count
print(f"Words in first review: {documents[0][0][:20]}") # First 20 words of first review
print(f"Label of first review: {documents[0][1]}")      # Label: 'pos' or 'neg'
import nltk
from nltk.corpus import movie_reviews
import random

nltk.download('movie_reviews')

# Prepare documents
documents = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]
random.shuffle(documents)

# Create frequency distribution of all words
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())

# Select top 2000 words as features
word_features = list(all_words)[:2000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Create feature sets
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Split into training and testing data (80% train, 20% test)
train_size = int(len(featuresets) * 0.8)
train_set = featuresets[:train_size]
test_set = featuresets[train_size:]

# Train Naive Bayes classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show the most informative features
classifier.show_most_informative_features(10)
