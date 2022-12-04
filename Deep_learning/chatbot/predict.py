import numpy as np
import tensorflow as tf
import pickle
import data_preprocess as df


def chat_predict():
    with open('chat_bot_model.pickle', 'rb') as chat_model:
        model = pickle.load(chat_model)

    with open('tokenizer.pickle', 'rb') as token:
        tokenizer = pickle.load(token)

    with open('label_encoder.pickle', 'rb') as ecn:
        encoder = pickle.load(ecn)

    while True:
        user_input = input('Enter your query:')
        if user_input.lower() == 'quit':
            break
        prediction = model.predict(
            tf.keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([user_input]),truncating='post',
                                                          maxlen=20))
        tag = encoder.inverse_transform([np.argmax(prediction)])
        for intent in df.data['intents']:
            if intent['tag'] == tag:
                response = np.random.choice(intent['responses'])
                print(f'Bot replies:{response}')



