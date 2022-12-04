from sklearn.preprocessing import LabelEncoder
import data_preprocess as dp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size = 1000
max_len = 20


def encoding():
    """label encoding of target variable"""
    lbl_enc = LabelEncoder()
    lbl_enc.fit(dp.training_labels)
    training_labels = lbl_enc.transform(dp.training_labels)
    return training_labels,lbl_enc


def tokenizer():
    oov_token = "<OOV>"
    tok_obj = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tok_obj.fit_on_texts(dp.training_sentences)
    word_index = tok_obj.word_index
    sequence = tok_obj.texts_to_sequences(dp.training_sentences)
    padded_sentence = pad_sequences(sequence,
                                    maxlen=max_len,
                                    truncating='post')
    return padded_sentence,tok_obj
