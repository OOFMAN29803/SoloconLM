import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import random

def sample_from_logits(logits, temperature=1.0):
    """ Apply temperature to logits and sample an index from the output probabilities. """
    scaled_logits = logits / temperature
    probabilities = tf.nn.softmax(scaled_logits).numpy()
    return np.random.choice(len(probabilities), p=probabilities)

# Load the model and tokenizer once, outside the loop
model = tf.keras.models.load_model('SoloconLM Beta 1.h5')
with open('tokenizerBeta1.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def generate_text(model, tokenizer, seed_text, num_words=50):
    """
    Generates text starting from a seed_text.
    model: The trained Keras model
    tokenizer: The tokenizer instance
    seed_text: The seed string to start text generation
    num_words: Number of words to generate
    """
    text_generated = seed_text
    for _ in range(num_words):
        # Convert the current text input into a sequence of integers
        encoded_text = tokenizer.texts_to_sequences([text_generated])
        # Pad sequences to the fixed length the model was trained with
        pad_encoded = pad_sequences(encoded_text, maxlen=100, truncating='pre')
        # Predict probabilities for each word
        pred_prob = model.predict(pad_encoded)
        pred_index = sample_from_logits(pred_prob[0], temperature=1.0)  # Sampling based on the output probabilities
        # Retrieve actual word from tokenizer
        pred_word = tokenizer.index_word.get(pred_index, '[UNK]')
        # Append the predicted word to the existing text
        text_generated += ' ' + pred_word
    return text_generated

while True:
    input_text = input("prompt: ")
    if not input_text.strip():
        print("Please enter some text.")
        continue
    generated_text = generate_text(model, tokenizer, input_text, num_words=random.randint(5,30))
    print("Generated text:", generated_text)
