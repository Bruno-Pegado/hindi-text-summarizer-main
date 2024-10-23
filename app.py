from flask import Flask, render_template, request
from src.utils.preprocess import preprocess
from src.tf_idf.td_idf import tf_idf
from src.utils.dictionaries import synonym_dict
from rouge import Rouge

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
        original_text = paragraph

        op = preprocess(paragraph)
        sentences = [' '.join(sentence) for sentence in op]

        tf_idf_scores = tf_idf(op)
        
        sentence_scores = []
        for idx, sentence_scores_dict in enumerate(tf_idf_scores):
            sentence_total_score = sum(sentence_scores_dict.values())
            sentence_scores.append((idx, sentence_total_score))
        
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        n = len(sentence_scores)
        a = round(n * 0.7)
        top_sentences_idx = [idx for idx, score in sentence_scores[:a]]
        
        top_sentences_idx.sort()
        
        summarized_text = " ".join([sentences[idx] for idx in top_sentences_idx])

        summarized_text = replace_synonyms(summarized_text)

        accuracy = calculate_accuracy(original_text, summarized_text)
        for rouge in accuracy.items():
            print(rouge)

    return render_template('index.html', original_text=original_text, summary_text=summarized_text, accuracy=accuracy)

def replace_synonyms(text: str) -> str:
    words = text.split()

    replaced_words = [synonym_dict.get(word, word) for word in words]

    return ' '.join(replaced_words)

def calculate_accuracy(original_text: str, summary_text: str) -> dict:
    rouge = Rouge()
    scores = rouge.get_scores(summary_text, original_text)
    rouge_1 = scores[0]['rouge-1']
    rouge_2 = scores[0]['rouge-2']
    rouge_l = scores[0]['rouge-l']
    return {
        "ROUGE-1": {
            "Precision": round(rouge_1['p'] * 100, 2),
            "Recall": round(rouge_1['r'] * 100, 2),
            "F1-Score": round(rouge_1['f'] * 100, 2),
        },
        "ROUGE-2": {
            "Precision": round(rouge_2['p'] * 100, 2),
            "Recall": round(rouge_2['r'] * 100, 2),
            "F1-Score": round(rouge_2['f'] * 100, 2),
        },
        "ROUGE-L": {
            "Precision": round(rouge_l['p'] * 100, 2),
            "Recall": round(rouge_l['r'] * 100, 2),
            "F1-Score": round(rouge_l['f'] * 100, 2),
        }
    }


if __name__ == '__main__':
    app.run(debug=True)
