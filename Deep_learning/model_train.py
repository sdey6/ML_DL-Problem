import label_encoding as le
import data_preprocess as dp
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
import numpy as np
import pickle

embedding_dim = 16
tr_labels,label_encd_obj = le.encoding()
tr_sentences,token_obj = le.tokenizer()
model = Sequential()
model.add(Embedding(le.vocab_size, embedding_dim, input_length=le.max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(dp.num_classes, activation='softmax'))
# for layer in model.layers:print(f'\n{layer.get_weights()}')
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(tr_sentences, np.array(tr_labels), epochs=100, verbose=False)

with open('./chat_bot_model.pickle', 'wb') as model_pck:
    pickle.dump(model, model_pck, protocol=pickle.HIGHEST_PROTOCOL)

with open('./tokenizer.pickle', 'wb') as token:
    pickle.dump(token_obj, token, protocol=pickle.HIGHEST_PROTOCOL)


with open('./label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(label_encd_obj, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)