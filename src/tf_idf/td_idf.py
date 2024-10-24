import math
from collections import defaultdict

def compute_tf(word_list):
    tf_dict = {}
    total_words = len(word_list)
    
    for word in word_list:
        tf_dict[word] = tf_dict.get(word, 0) + 1
    
    for word in tf_dict:
        tf_dict[word] /= total_words
    
    return tf_dict

def compute_idf(processed_text):
    num_sentences = len(processed_text)
    idf_dict = defaultdict(int)
    
    for sentence in processed_text:
        for word in set(sentence):
            if word:
                idf_dict[word] += 1
    
    for word in idf_dict:
        idf_dict[word] = math.log(num_sentences / float(idf_dict[word]))
    
    return idf_dict

def compute_tf_idf(tf_dict, idf_dict):
    tf_idf_dict = {}
    
    for word, tf_value in tf_dict.items():
        tf_idf_dict[word] = tf_value * idf_dict.get(word, 0)
    
    return tf_idf_dict

def tf_idf(processed_text):
    idf_dict = compute_idf(processed_text)
    
    tf_idf_scores = []
    for sentence in processed_text:
        tf_dict = compute_tf(sentence)
        tf_idf_dict = compute_tf_idf(tf_dict, idf_dict)
        tf_idf_scores.append(tf_idf_dict)
    
    return tf_idf_scores