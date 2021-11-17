import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd

###############################################

print("Imported")

filename = 'spotify_dataset2.csv'

df = pd.read_csv(filename, low_memory=False)

print("Dataframe: ")

print(df)
df.describe()

print("====================================")

row = df.iloc[0]

print("Row: ")
print(row)

print("TEST: ")
print(df.iloc[42000])

print("====================================")

genreEncoder = LabelEncoder()
df['genre'] = genreEncoder.fit_transform(df['genre'])
numGenres = len(genreEncoder.classes_)
print(len(genreEncoder.classes_), genreEncoder.classes_)
df['mode'] = genreEncoder.fit_transform(df['mode'])
#df['key'] = LabelEncoder().fit_transform(df['key'])
#df['time_signature'] = LabelEncoder().fit_transform(df['time_signature'])


print("====================================")

# Normalize Data
normalizeColumns = ['popularity', 'loudness', 'tempo', 'duration_ms']
#normalizeColumns = ['loudness', 'tempo', 'duration_ms', 'danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
df[normalizeColumns] = StandardScaler().fit_transform(df[normalizeColumns])

print("====================================")

print(df.describe())

print("====================================")

print(df.head())

print("====================================")

regularColumns = ['popularity', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
#regularColumns = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
oneHotVectorColumns = ['key', 'time_signature']
outputColumn = ['genre']

filtered = df[regularColumns].to_numpy()

print(filtered.shape)

print("====================================")

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
encoded = enc.fit_transform(df[oneHotVectorColumns])

print(enc.categories_)

print(df[oneHotVectorColumns])
print(encoded.shape)
print(encoded[0, :])

X = np.concatenate((filtered, encoded), axis=1)
print(X.shape)
y = df[outputColumn].to_numpy()
print(y.shape)


print("====================================")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

print("====================================")
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


model = RandomForestClassifier(max_depth=30, random_state=0)
model.fit(X_train, y_train)

print("Train Accuracy: ", accuracy_score(y_train, model.predict(X_train)))

y_predicted = model.predict(X_test)
print("Test Accuracy: ", accuracy_score(y_test, y_predicted))


clf = DecisionTreeClassifier(max_depth=30, min_samples_split=10, random_state=42)
clf.fit(X_train, y_train)

print("Train Accuracy: ", accuracy_score(y_train, clf.predict(X_train)))

y_predicted = clf.predict(X_test)
print("Test Accuracy: ", accuracy_score(y_test, y_predicted))

print("====================================")
'''
_, inputDim = X.shape
outputDim = numGenres

model = Sequential()
# Purposely try to overfit
model.add(Dense(120, input_dim = inputDim, activation = 'relu')) # Rectified Linear Unit Activation Function
#model.add(Dense(90, activation = 'relu'))
#model.add(Dense(90, activation = 'relu'))
model.add(Dense(120, activation = 'relu'))
model.add(Dense(outputDim, activation = 'softmax')) # Softmax for multi-class classification
# Compile model here
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

y_pred = model.predict(X_test).argmax(axis=1)
print(y_pred.shape, y_pred)
print(y_test.shape, y_test)

from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)

import sys
np.set_printoptions(threshold=sys.maxsize)
print(matrix)


'''

n_features = X_train.shape[1]
# define model
model = Sequential()
	model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=1)
# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)

'''









