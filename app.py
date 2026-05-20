from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import re

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# 🔥 Load Model
model = load_model("model_clean.h5", compile=False)

# 🔥 Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

max_text_len = 20
max_summary_len = 10


# 🧹 Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text


# 🔁 Decode sequence → words
def seq_to_text(seq):
    words = []
    for idx in seq:
        word = reverse_word_index.get(idx, "")
        if word not in ["startseq", "endseq", ""]:
            words.append(word)
    return " ".join(words)


# 🧠 Generate Summary (Step-by-step decoding)
def generate_summary(input_text):

    input_text = clean_text(input_text)

    seq = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(seq, maxlen=max_text_len, padding='post')

    summary = "startseq"

    for _ in range(max_summary_len):
        seq_summary = tokenizer.texts_to_sequences([summary])
        seq_summary = pad_sequences(seq_summary, maxlen=max_summary_len-1, padding='post')

        prediction = model.predict([padded, seq_summary], verbose=0)

        predicted_index = np.argmax(prediction[0, -1, :])
        predicted_word = reverse_word_index.get(predicted_index, "")

        if predicted_word == "endseq" or predicted_word == "":
            break

        summary += " " + predicted_word

    return summary.replace("startseq", "").strip()


# 🌐 Home
@app.route('/')
def home():
    return render_template("index.html")


# 📡 API
@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json
    text = data.get("text")

    summary = generate_summary(text)

    return jsonify({"summary": summary})


# ▶️ Run
if __name__ == "__main__":
    app.run(debug=True)