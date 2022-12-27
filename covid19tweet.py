import pandas as pd
import re
import nltk
nltk.download('punkt')
import string
from pathlib import Path
RESULTS_DIR = 'drive/MyDrive'
covid19_tweets = pd.read_csv(RESULTS_DIR +'/covid19_tweets.csv')


# Concatenate the papers from different sources into one corpus dataframe.
corpus_df = pd.concat([covid19_tweets])

def clean_sentence(sentence):
    """
    Cleans an individual sentence.
    
    Args:
        sentence: An individual sentence from the corpus.
    Returns:
        clean_sentence: A cleaned sentence with unwanted linespaces removed.
    """
    
    # Remove titles before linespaces
    try:
        clean_sentence = re.findall('\n\n(.*)', sentence)[-1]
    except IndexError:
        clean_sentence = sentence
    
    return clean_sentence

def clean_sentences(sentences):
    """
    Cleans each sentence in a list of sentences, corresponding to one article.
    
    Args:
        sentences: A list of sentences from an article (paper).
    Returns:
        cleaned_sentences_no_punctuation: A list of cleaned sentences, with "sentences" that only contain punctuation removed.
    """
    
    cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]
    
    # Remove sentences with only punctuation
    cleaned_sentences_no_punctuation = [sentence for sentence in cleaned_sentences if sentence not in string.punctuation]
    
    return cleaned_sentences_no_punctuation
    
    
# Retain the paper ID throughout the analysis.
# The paper ID can be re-used to refer back to the relevant paper.
corpus = corpus_df[['user_name', 'text']].values
corpus = [(user_name, document) for user_name, document in corpus if type(document)==str]

# Split into sentences for each article in the corpus
documents = [(user_name, nltk.sent_tokenize(document)) for user_name, document in corpus]

# Clean each sentence
documents_clean = [(user_name, clean_sentences(document)) for user_name, document in documents]
        
# Flatten documents into a list of sentences
sentences_clean = [item for sublist in documents_clean for item in sublist]