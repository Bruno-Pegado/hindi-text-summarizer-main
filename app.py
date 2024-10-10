from flask import Flask, render_template, request
import time
from src.utils.preprocess import preprocess
from src.tf_idf.td_idf import tf_idf
from src.utils.dictionaries import synonym_dict

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    paragraph = request.form['hindi_text']
    summarized_text = ""
    original_text = ""

    if not paragraph.strip():
        original_text = "No text provided"
    else:
        original_text = paragraph  # Store original text for display
        start_time = time.time()

        op = preprocess(paragraph)
        sentences = [' '.join(sentence) for sentence in op]

        tf_idf_scores = tf_idf(op)
        
        sentence_scores = []
        for idx, sentence_scores_dict in enumerate(tf_idf_scores):
            sentence_total_score = sum(sentence_scores_dict.values())
            sentence_scores.append((idx, sentence_total_score))
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        n = len(sentence_scores)
        a = round(n/2)
        top_sentences_idx = [idx for idx, score in sentence_scores[:a]]
        
        top_sentences_idx.sort()
        
        summarized_text = " ".join([sentences[idx] for idx in top_sentences_idx])

        summarized_text = replace_synonyms(summarized_text)
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        execution_time = f"The execution time is: {elapsed_time:.4f} seconds"

    return render_template('index.html', original_text=original_text, summary_text=summarized_text)

def replace_synonyms(text: str) -> str:
    words = text.split()

    replaced_words = [synonym_dict.get(word, word) for word in words]

    return ' '.join(replaced_words)

if __name__ == '__main__':
    app.run(debug=True)
