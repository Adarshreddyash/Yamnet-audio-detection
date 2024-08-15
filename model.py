# !pip install gTTS

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from gtts import gTTS
import IPython.display as ipd
import os

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_yamnet_embeddings(audio_url, filename):
    os.system(f"wget -O {filename} {audio_url}")
    
    audio, sr = librosa.load(filename, sr=16000)
    
    scores, embeddings, spectrogram = yamnet_model(audio)
    
    os.remove(filename)
    
    return np.mean(embeddings, axis=0)

audio_files_and_labels = [
    ('https://github.com/Adarshreddyash/sounds-datasets/raw/main/ttsmaker-file-2024-8-15-14-51-28.wav', 'eehh'),
    ('https://github.com/Adarshreddyash/sounds-datasets/raw/main/ttsmaker-file-2024-8-15-14-50-22.wav', 'aahh')
]

embeddings = []
labels = []

for audio_url, label in audio_files_and_labels:
    filename = audio_url.split('/')[-1]
    embedding = extract_yamnet_embeddings(audio_url, filename)
    embeddings.append(embedding)
    labels.append(label)

X = np.array(embeddings)
y = np.array(labels)

print("Features (X):", X)
print("Labels (y):", y)

# Encode labels (e.g., "aaaaa" -> 0, "eeeee" -> 1)
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

def predict_custom_audio(model, audio_url):
    filename = audio_url.split('/')[-1]
    embedding = extract_yamnet_embeddings(audio_url, filename)
    embedding = np.expand_dims(embedding, axis=0)
    prediction = model.predict(embedding)
    predicted_label = le.inverse_transform([np.argmax(prediction)])
    
    # Convert text to speech
    tts = gTTS(predicted_label[0], lang='en')
    tts.save('output_speech.wav')
    
    return predicted_label[0], 'output_speech.wav', np.max(prediction)

text, audio_file, score = predict_custom_audio(model, 'https://github.com/Adarshreddyash/sounds-datasets/raw/main/audio_data/test_audio.wav')
print(f"Predicted text: {text}")
print(f"Score: {score}")
ipd.display(ipd.Audio(audio_file))
