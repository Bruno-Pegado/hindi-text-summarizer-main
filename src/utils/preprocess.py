import pandas as pd
from indicnlp.tokenize import sentence_tokenize
from indicnlp.tokenize import indic_tokenize
import re
import string
from src.utils.dictionaries import lemmatization_dict,stopwords_mr



def preprocess(paragraph:str):
    # sentence-tokeniation
    tokenized_sentences = sentence_tokenize.sentence_split(paragraph, lang='hi')

    # word-tokenization
    tokenized_words = word_tokenize(tokenized_sentences)

    # lemmatization
    lemmatized_word_op = lemmatize_tokenized_words(tokenized_words)

    # stop words removal
    stopword_removed = remove_stopwords(lemmatized_word_op)

    # filtration
    filtration_op = filter_hindi(stopword_removed)

    # scrip validation
    punc_processed = remove_punctuation(filtration_op)
    return punc_processed



def word_tokenize(sentences: list):
    # Tokenize each sentence into words
    word_tokenized_list = []
    
    for sentence in sentences:
        output = list(indic_tokenize.trivial_tokenize(sentence, lang='hi'))
        word_tokenized_list.append(output)
    return word_tokenized_list



def lemmatize_word(word: str) -> str:
    return lemmatization_dict.get(word, word)  # Return the lemma or the word itself if not found

def lemmatize_tokenized_words(tokenized_words_list: list) -> list:
    lemmatized_list = []
    for sentence in tokenized_words_list:
        lemmatized_sentence = [lemmatize_word(word) for word in sentence]
        lemmatized_list.append(lemmatized_sentence)
    return lemmatized_list



def remove_stopwords(lemmatized_words_list: list) -> list:
    stopwords_removed = []
    for sentence in lemmatized_words_list:  # Iterate through lemmatized sentences
        # Remove stopwords from each sentence
        filtered_sentence = [word for word in sentence if word not in stopwords_mr]
        stopwords_removed.append(filtered_sentence)
    return stopwords_removed



def filter_hindi(tokenized_words_list):
    # Regular expression pattern for English words (to remove non-Devanagari words)
    english_pattern = re.compile(r'\b[a-zA-Z]+\b')

    # Filter each sentence in the tokenized list
    filtered_list = []
    for sentence in tokenized_words_list:
        filtered_sentence = [word for word in sentence if not english_pattern.fullmatch(word)]
        filtered_list.append(filtered_sentence)

    return filtered_list



def remove_punctuation(filtrated_words_list):
    # Create a translator to remove punctuation
    translator = str.maketrans('', '', string.punctuation)

    punc_removed_list = []
    for sentence in filtrated_words_list:
        punc_removed_sentence = [word.translate(translator) for word in sentence]
        punc_removed_list.append(punc_removed_sentence)

    return punc_removed_list
