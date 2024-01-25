import pickle
import numpy as np

# The Skipgram with negative sampling model is selected.
# Load its embedding from the pickle file.
Glove_embeddings = pickle.load(open('Glove_embeddings.pickle', 'rb'))

# Define the function to get the dot product
def dot_product(A, B):
    return np.dot(A, B)

# Define the function to get the most similar word (Later used in Web App)
def get_most_similar_word(embeddings, word, topn=10):
    similarities = {}
    word = word.lower()

    if word not in embeddings.keys():
        word_vec = np.array(embeddings['<UNK>'])
    else:
        word_vec = np.array(embeddings[word])

    for vocab, emb in embeddings.items():
        similarities[vocab] = dot_product(word_vec, np.array(emb))
    similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return similarities[1:topn + 1]
