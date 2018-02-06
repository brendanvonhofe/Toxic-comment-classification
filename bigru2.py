import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU, Input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

# the set classes to predict from
CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# for K-fold cross validation / ensemble-ish
PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4

# Import train data, 159570 samples
# Make data file-paths a command-line argument
train_data = pd.read_csv("data/train.csv")
text_data = train_data[['comment_text']].values.tolist()
labels = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
texts = [text[0] for text in text_data]

# import test data
test = pd.read_csv("data/test.csv")
test_data = test[['comment_text']].values.tolist()
test_ids = test[['id']].values
test_texts = [text[0] for text in test_data]

# Tokenizing raw text data
maxlen = 500 # HYPERPARAMETER -- length of text sequences to feed to the recurrent net
training_samples = 127656 # 80% of samples
validation_samples = 31914 # 20%
# max_words = 10000 # HYPERPARAMETER - glove.6B has 400k vocab size

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# Padding text sequences to all have the same length
data = pad_sequences(sequences, maxlen=maxlen)
data_test = pad_sequences(test_sequences, maxlen=maxlen)

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# # Shuffle dataset
# indices = np.arange(data.shape[0])
# np.random.shuffle(indices)
# data = data[indices]
# labels = labels[indices]

# # Train/Validation Split
# x_train = data[:training_samples]
# y_train = labels[:training_samples]
# x_val = data[training_samples: training_samples + validation_samples]
# y_val = labels[training_samples: training_samples + validation_samples]

x_train = data
y_train = labels

# Import Glove pre-trained word embeddings
# glove_dir = '/Users/brendan/Desktop/kaggle/toxic_comment_classification/bi_gru/data/glove.6B'

# HYPERPARAMETER - which pretrained embedding to use, different dimensionality
embeddings_index = {}
num_value_errors = 0
f = open("data/glove.840B.300d.txt")
# line = f.readline() # For Crawl embedding
for line in f:
	values = line.split()
	word = values[0]
	try:
		coefs = np.asarray(values[1:], dtype='float32')
	except ValueError:
		# print("Value Error... should maybe handle better than just to skip it")
		# print(line)
		num_value_errors+=1
	embeddings_index[word] = coefs

f.close()
print("Number of value errors [Glove tokens that can't be used] ", num_value_errors)

embedding_dim = 300 # depends on the pre-trained embedding thats being used [usually in the name]

# Embedding matrix -- Embedding Layer Weights
embedding_matrix = np.zeros((len(tokenizer.word_counts), embedding_dim))
for word, i in word_index.items():
	if i < len(tokenizer.word_counts):
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector


# Defining model

# HYPERPARAMETERS
recurrent_units = 64
dropout_rate = 0.3
dense_size = 32
batch_size = 256
num_folds = 10

kf = KFold(num_folds)

fold = 0
test_predicts_list = []

for train, test in kf.split(x_train, y_train):
	model = Sequential()
	model.add(Embedding(len(tokenizer.word_counts), 
		embedding_dim,
		input_length=maxlen, 
		weights=[embedding_matrix], 
		trainable=False))
	model.add(Bidirectional(CuDNNGRU(recurrent_units, return_sequences=True)))
	model.add(Dropout(dropout_rate))
	model.add(Bidirectional(CuDNNGRU(recurrent_units, return_sequences=False)))
	model.add(Dense(dense_size, activation='relu'))
	model.add(Dense(6, activation='sigmoid'))
	model.summary()

	model.compile(optimizer=RMSprop(clipvalue=1, clipnorm=1), 
		loss='binary_crossentropy', 
		metrics=['acc'])

	# Callbacks
	checkpointer = ModelCheckpoint(filepath="weights.fold."+str(fold)+".hdf5", 
		save_best_only=True,
		save_weights_only=True,
		monitor='val_loss',
		verbose=1)
	earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

	# train the model
	history = model.fit(x_train[train], 
		y_train[train], 
		epochs=100, 
		batch_size=batch_size, 
		validation_data=(x_train[test], y_train[test]),
		callbacks=[earlystopper, checkpointer])

	model.load_weights(filepath="weights.fold."+str(fold)+".hdf5", by_name=False)

	predictions = model.predict(data_test, batch_size=batch_size)
	test_predicts_list.append(predictions)

	fold += 1

	K.clear_session()

# # Display learning curves
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

# plt.show()

# model.load_weights(filepath="weights.06-0.04.hdf5", by_name=False)

print("Making predictions")
test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
	test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))
test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

test_ids = test_ids.reshape((len(test_ids), 1))
test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
test_predicts["id"] = test_ids
test_predicts = test_predicts[["id"] + CLASSES]
test_predicts.to_csv("submission.csv", index=False)

