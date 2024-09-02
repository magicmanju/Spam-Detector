# Installing Packages

""" 
!pip install transformers
!pip install keras_nlp
!pip install dash """

## Importing libraries

import numpy as np
import pandas as pd
import nltk, re, collections, pickle, os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

## Reading the Dataset and Data Manipulation

pd.set_option("display.precision", 3)
pd.options.display.float_format = '{:.3f}'.format
df = pd.read_csv('spam.csv', encoding = 'latin-1')
df = df.filter(['v1', 'v2'], axis = 1)
df.columns = ['feature', 'message']
df.drop_duplicates(inplace = True, ignore_index = True)
print('DataFrame Info:\n')
print(df.info())
print('\nNumber of null values:\n')
print(df.isnull().sum())

##  Spam Classification using Machine Learning

##Text processing

vocab_size = 1000
embed_dim = 64
oov_token = "<OOV>"
test_size, valid_size = 0.05, 0.2
num_epochs = 20
drop_level = 0.3
trunc_type = 'post'
padding_type = 'post'
threshold = 0.5
seed = 42
preprocessed_df = []
lemmatizer = WordNetLemmatizer()
for i in range(df.shape[0]):
    message = df.iloc[i, 1]
    message = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', message)
    message = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', message)
    message = re.sub('£|\$', 'moneysymb', message)
    message = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', message)
    message = re.sub('\d+(\.\d+)?', 'numbr', message)
    message = re.sub('[^\w\d\s]', ' ', message)
    message = re.sub('[^A-Za-z]', ' ', message).lower()
    token_messages = word_tokenize(message)
    mess = []
    for word in token_messages:
        if word not in set(stopwords.words('english')):
            mess.append(lemmatizer.lemmatize(word))
    txt_mess = " ".join(mess)
    preprocessed_df.append(txt_mess)
print("Text Processed")

##Vectorization and dumping the vectorization

count_vectorizer = CountVectorizer(max_features=vocab_size)
X_count = count_vectorizer.fit_transform(preprocessed_df)
X_count_array = X_count.toarray()
y = df['feature']
X_train, X_test, y_train, y_test = train_test_split(
    X_count_array, y, test_size=(test_size + valid_size), random_state=seed
)

print(f'Number of rows in test set: {X_test.shape[0]}')
print(f'Number of rows in training set: {X_train.shape[0]}')

tfidf_vectorizer = TfidfVectorizer(max_features=vocab_size)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_df)

tfidf_data = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

with open('TF_IDF.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)

print("TF-IDF vectorizer saved to 'TF_IDF.pkl'")

##Classification models

##Multinomial Naive Bayes

class_MNB = MultinomialNB().fit(X_train, y_train)
y_pred_MNB = class_MNB.predict(X_test)
class_rep_MNB = classification_report(y_test, y_pred_MNB)
print('\t\t\tClassification report:\n\n', class_rep_MNB, '\n')

###  Decision Tree Classifier.

class_DTC = DecisionTreeClassifier(random_state = seed).fit(X_train, y_train)
y_pred_DTC = class_DTC.predict(X_test)
class_rep_DTC = classification_report(y_test, y_pred_DTC)
print('\t\t\tClassification report:\n\n', class_rep_DTC, '\n')

###Logistic Regression.

class_LR = LogisticRegression(random_state = seed, solver = 'liblinear').fit(X_train, y_train)
y_pred_LR = class_LR.predict(X_test)
class_rep_LR = classification_report(y_test, y_pred_LR)
print('\t\t\tClassification report:\n\n', class_rep_LR, '\n')

###  KNeighbors Classifier.

class_KNC = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
y_pred_KNC = class_KNC.predict(X_test)
class_rep_KNC = classification_report(y_test, y_pred_KNC)
print('\t\t\tClassification report:\n\n', class_rep_KNC, '\n')

###  Support Vector Classification.

class_SVC = SVC(probability = True, random_state = seed).fit(X_train, y_train)
y_pred_SVC = class_SVC.predict(X_test)
class_rep_SVC = classification_report(y_test, y_pred_SVC)
print('\t\t\tClassification report:\n\n', class_rep_SVC, '\n')

## Gradient Boosting Classifier.

class_GBC = GradientBoostingClassifier(random_state = seed).fit(X_train, y_train)
y_pred_GBC = class_GBC.predict(X_test)
class_rep_GBC = classification_report(y_test, y_pred_GBC)
print('\t\t\tClassification report:\n\n', class_rep_GBC, '\n')

## Spam Detection using Deep Learning

from tensorflow import keras
from tensorflow.keras import layers
import keras.layers
import keras_nlp
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay

##Label Transformation.

sentences_new_set = df['message'].values
labels_new_set = df['feature'].values
train_size = int(df.shape[0] * (1 - test_size - valid_size))
valid_bound = int(df.shape[0] * (1 - valid_size))
train_sentences = sentences_new_set[0 : train_size]
valid_sentences = sentences_new_set[train_size : valid_bound]
test_sentences = sentences_new_set[valid_bound : ]
train_labels_str = labels_new_set[0 : train_size]
valid_labels_str = labels_new_set[train_size : valid_bound]
test_labels_str = labels_new_set[valid_bound : ]
train_labels = [1 if item == 'ham' else 0 for item in train_labels_str]
valid_labels = [1 if item == 'ham' else 0 for item in valid_labels_str]
test_labels = [1 if item == 'ham' else 0 for item in test_labels_str]
train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)

##  Tokenization.

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token, lower=False)
tokenizer.fit_on_texts(train_sentences)
max_len = 250
padding_type = 'post'
trunc_type = 'post'

def preprocess_texts(tokenizer, texts, max_len, padding_type, trunc_type):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, padding=padding_type, maxlen=max_len, truncating=trunc_type)

train_set = preprocess_texts(tokenizer, train_sentences, max_len, padding_type, trunc_type)
valid_set = preprocess_texts(tokenizer, valid_sentences, max_len, padding_type, trunc_type)
test_set = preprocess_texts(tokenizer, test_sentences, max_len, padding_type, trunc_type)

size_voc = len(tokenizer.word_index) + 1

print(f"Training set shape: {train_set.shape}")
print(f"Validation set shape: {valid_set.shape}")
print(f"Test set shape: {test_set.shape}")
print(f"Vocabulary size: {size_voc}")

##  Model building.

inputs = keras.layers.Input(shape=(max_len,), dtype=tf.int32)
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(size_voc, max_len, embed_dim)(inputs)
decoder = keras_nlp.layers.TransformerDecoder(intermediate_dim=embed_dim,
                                                            num_heads=8,
                                                            dropout=0.3)(embedding_layer)
gru = layers.Bidirectional(layers.GRU(128, return_sequences=True))(decoder)
avg_pool = layers.GlobalAveragePooling1D()(gru)
max_pool = layers.GlobalMaxPool1D()(gru)
x = layers.Concatenate()([avg_pool, max_pool])
outputs = keras.layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

##  Model compiling and fitting.

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=1000,
    decay_rate=0.96
)
optim = Adam(learning_rate=lr_schedule)
model.compile(loss = 'binary_crossentropy',
              optimizer = optim,
              metrics = ['accuracy'])
model.summary()

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
model_checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)
history = model.fit(
    train_set,
    train_labels,
    epochs=20,
    validation_data=(valid_set, valid_labels),
    verbose=1,
    callbacks=[early_stopping, model_checkpoint]
)

##Results visualization.

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.show()

batch_size = 32
model_score = model.evaluate(test_set, test_labels, batch_size=batch_size, verbose=1)
loss = model_score[0]
accuracy = model_score[1]
print(f"Test accuracy: {accuracy * 100:0.2f}% \t\t Test error: {loss:0.4f}")

##Model saving and predict checking.

from tensorflow.keras.models import load_model
model.save('model1.keras')
threshold = 0.5
y_pred = model.predict(test_set)
y_prediction = (y_pred > threshold).astype(int)
conf_m = confusion_matrix(test_labels, y_prediction)
class_rep = classification_report(test_labels, y_prediction)

print('\t\t\tClassification report:\n\n', class_rep, '\n')

##Results

from dash import dcc, html, Input, Output, Dash
import plotly.graph_objects as go
app = Dash(__name__)
app.layout = html.Div([
    dcc.Input(id='input-message', type='text', value=input("Enter the message:")),
    html.Button('Classify', id='classify-button', n_clicks=0),
    html.Div(id='result-text'),
    dcc.Graph(id='result-chart')
])

@app.callback(
    [Output('result-chart', 'figure'),
     Output('result-text', 'children')],
    [Input('classify-button', 'n_clicks')],
    [Input('input-message', 'value')]
)
def update_graph(n_clicks, input_message):
    if n_clicks > 0:
        message_tp = pad_sequences(
            tokenizer.texts_to_sequences([input_message]),
            maxlen=max_len,
            padding=padding_type,
            truncating=trunc_type
        )
        pred = model.predict(message_tp)[0][0]
        threshold = 0.5
        classification = "Real Text" if pred > threshold else "Spam Message"

        labels = ['Real Text', 'Spam Message']
        values = [pred, 1 - pred]
        colors = ['green', 'red']
        fig = go.Figure(data=[
            go.Bar(name='Probability', x=labels, y=values, marker_color=colors)
        ])

        fig.update_layout(
            title='Message Classification Probability',
            xaxis_title='Category',
            yaxis_title='Probability',
            yaxis=dict(range=[0, 1]),
            template='plotly_dark'
        )
        fig.add_annotation(
            text=f"Classification: {classification}",
            x=0.5,
            y=1.05,
            showarrow=False,
            xref='paper',
            yref='paper',
            font=dict(size=16, color='black'),
            align='center'
        )
        return fig, f"This message is classified as: {classification}"
    return go.Figure(), "Please enter a message and click 'Classify'"

if __name__ == '__main__':
    app.run_server(debug=True)