import os
import pickle
import numpy as np
from rdkit import Chem
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec

try:
    model_file_path = os.path.join('solution_coding', 'model_300dim.pkl')
    w2v_model = word2vec.Word2Vec.load(model_file_path)

except FileNotFoundError as e:
    print(f"Error loading the word2vec model: {e}")

try:
    def sentences2vec(sentences, model, unseen=None):

        keys = set(model.wv.key_to_index.keys())
        if unseen is None:
            unseen = np.zeros(model.vector_size, dtype=float)

        vectors = []
        for sentence in sentences:
            vec = np.sum([model.wv.get_vector(word) for word in sentence if word in keys], axis=0)
            if len(vec) == 0:
                vec = unseen
            vectors.append(vec)

        return np.array(vectors)

except Exception as e:
    print(f"An unexpected error occurred in sentences2vec function: {e}")

try:
    def generate_word_embedding(smiles_example):

        mol = Chem.MolFromSmiles(smiles_example)
        sentence = MolSentence(mol2alt_sentence(mol, radius=1))
        word_embedding = sentences2vec([sentence], w2v_model)
        result = np.array(word_embedding)

        return result

except Exception as e:
    print(f"An unexpected error occurred in word_embedding function: {e}")

try:
    def generate_word_embeddings(smiles_example):
        mol = Chem.MolFromSmiles(smiles_example)
        sentence = MolSentence(mol2alt_sentence(mol, radius=1))
        word_embedding = sentences2vec([sentence], w2v_model)

        return word_embedding[0]  # Take the first element to convert it to a 2D array

except Exception as e:
    print(f"An unexpected error occurred in word_embeddings function: {e}")


try:
    def preprocessing_we(target, assay_type, data):
               
        print("Starting molecules embedding: ")
        w_embedings = data["canonical_smiles"].apply(generate_word_embeddings).tolist()

        return w_embedings
    
except Exception as e:
    print(f"An unexpected error occurred in df word_embeddings function: {e}")
