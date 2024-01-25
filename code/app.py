from flask import Flask, render_template, request
from function import get_most_similar_word
from function import Glove_embeddings
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    most_similar_words = None
    if request.method == 'POST':
        search_word = request.form.get('search')
        if search_word:
            most_similar_words = get_most_similar_word(Glove_embeddings, search_word)

    return render_template('index.html', most_similar_words=most_similar_words)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
