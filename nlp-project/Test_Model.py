# -*- coding: utf-8 -*-
import re
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import silhouette_score
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import numpy as np
import nltk

def download_nltk_packages():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet = True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet = True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet = True)

def preprocess_text(text, stop_words, lemmatizer):
    """
    Preprocess a single text document using the same steps as training
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokenized = word_tokenize(text)
    tokenized = [word for word in tokenized if word not in stop_words]
    lemmatized = [lemmatizer.lemmatize(word) for word in tokenized]
    return ' '.join(lemmatized)

def load_model_and_vectorizer():
    """
    Load the saved NMF model and TF-IDF vectorizer
    """
    with open('nmf_model.pkl', 'rb') as f:
        nmf = pickle.load(f)
    
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidfVectorizer = pickle.load(f)
    
    return nmf, tfidfVectorizer

def evaluate_new_data(file_path):
    """
    Evaluate new data using the saved model and preprocessing
    """
    # Load the saved model and vectorizer
    nmf, tfidfVectorizer = load_model_and_vectorizer()
    
    # Load new data (assuming same CSV format as training)
    try:
        article = pd.read_csv(file_path)
        if 'content' not in article.columns:
            raise ValueError("CSV file must contain a 'content' column")
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Initialize preprocessing tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Preprocess the new data
    clean_content = []
    for corpus in article['content']:
        clean_content.append(preprocess_text(corpus, stop_words, lemmatizer))
    
    # Transform using the saved vectorizer
    X_new_tfidf = tfidfVectorizer.transform(clean_content)
    
    # Get topic distributions
    docTopic = nmf.transform(X_new_tfidf)
    
    # Get dominant topics
    dominant_topics = np.argmax(docTopic, axis=1)
    
    # Calculate silhouette score
    sil_score = silhouette_score(X_new_tfidf, dominant_topics, metric = 'cosine')
    
    # Print topic distribution
    topicCounts = {}
    for topic in dominant_topics:
        if topic not in topicCounts:
            topicCounts[topic] = 1
        else:
            topicCounts[topic] += 1
    
    print("\nTopic distribution in new data:")
    for topic, count in sorted(topicCounts.items()):
        print(f"Topic {topic}: {count} documents")
    
    # Print top words for each topic (from the trained model)
    featureNames = tfidfVectorizer.get_feature_names_out()
    nTopWords = 4
    
    print("\nTop words for each topic:")
    for topicIdx, topic in enumerate(nmf.components_):
        topWordIdx = topic.argsort()[::-1][:nTopWords]
        topWords = [featureNames[i] for i in topWordIdx]
        print(f"Topic {topicIdx}: {', '.join(topWords)}")

    # Tokenize (not join) for coherence model
    tokenized_docs = [corpus.split() for corpus in clean_content]

    # Create dictionary and corpus for gensim
    dictionary = Dictionary(tokenized_docs)
    corpus_gensim = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    # Get top words for each topic (same as what you print)
    nTopWords = 10
    featureNames = tfidfVectorizer.get_feature_names_out()
    topics = []
    for topic in nmf.components_:
        topIndices = topic.argsort()[::-1][:nTopWords]
        topicWords = [featureNames[i] for i in topIndices]
        topics.append(topicWords)

    # Compute Coherence Score
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_docs,
        dictionary=dictionary,
        coherence='c_v'
    )

    coh_score = coherence_model.get_coherence()

    print("\nSilhouette Score on new data:", round(sil_score, 2), "\n")
    print("\nCoherence Score on new data:", round(coh_score, 2), "\n")

if __name__ == "__main__":
    # Run evaluation
    download_nltk_packages()
    evaluate_new_data("C:\\Users\\negam\\Downloads\\articles1.csv\\articles1.csv")