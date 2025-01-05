from flask import Flask, request, jsonify
from flask_wtf.csrf import CSRFProtect
from flask_httpauth import HTTPTokenAuth
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from langdetect import detect
import csv
import os
import secrets
import bleach

# Flask-Initialisierung
app = Flask(__name__)

# CSRF-Schutz aktivieren
csrf = CSRFProtect(app)

# Authentifizierung (API Token)
auth = HTTPTokenAuth(scheme='Bearer')

# Ratenbegrenzung (Begrenzung auf 100 Anfragen pro Stunde)
limiter = Limiter(app, key_func=get_remote_address)

# API-Token für Authentifizierung (hier ein zufällig generiertes Beispiel)
API_TOKENS = {
    os.getenv("API_TOKEN"): 'admin'  # Sicherer Umgang mit Umgebungsvariablen
}

# *** Sicherheitsmaßnahmen: Authentifizierung ***
@auth.verify_token
def verify_token(token):
    if token in API_TOKENS:
        return API_TOKENS[token]
    return None

# *** Regelbasierte Wissensbasis ***
knowledge_base = {
    "hallo": "Hallo! Wie kann ich Ihnen helfen?",
    "wie geht es dir": "Mir geht es gut, danke der Nachfrage! Wie kann ich helfen?",
    "was kannst du": "Ich bin ein einfacher KI-Chatbot. Stellen Sie mir Fragen oder sagen Sie 'Hilfe', um mehr zu erfahren.",
    "hilfe": "Ich kann grundlegende Fragen beantworten. Zum Beispiel: 'Wie geht es dir?' oder 'Was kannst du?'",
    "auf wiedersehen": "Auf Wiedersehen! Bis zum nächsten Mal."
}

# *** Charakterverwaltung ***
characters = {}

def create_character(name, description):
    """Erstellt einen neuen Charakter."""
    if name in characters:
        return f"Ein Charakter mit dem Namen {name} existiert bereits."
    characters[name] = {
        "description": description,
        "conversation_history": []
    }
    return f"Charakter '{name}' wurde erfolgreich erstellt."

def select_character(name):
    """Wechselt zu einem bestehenden Charakter."""
    if name not in characters:
        return f"Charakter '{name}' existiert nicht. Bitte erstellen Sie ihn zuerst."
    return f"Charakter '{name}' wurde ausgewählt. Beschreibung: {characters[name]['description']}"

def character_response(character_name, user_input):
    """Generiert eine Antwort basierend auf dem Charakter."""
    if character_name not in characters:
        return "Kein Charakter ausgewählt oder Charakter existiert nicht."
    
    # Bereinigung der Benutzereingabe
    user_input = bleach.clean(user_input)

    # Füge den Dialog zur Gesprächshistorie hinzu
    characters[character_name]['conversation_history'].append({"user": user_input})
    
    # Antwort generieren
    response = f"{character_name}: Ich bin nicht sicher, wie ich darauf antworten soll. Kannst du mir mehr erzählen?"
    
    # Dialog in die Historie hinzufügen
    characters[character_name]['conversation_history'][-1]["bot"] = response
    return response

# *** Trainingsdaten erstellen ***
def generate_training_data():
    inputs = [
        "Hallo", "Guten Tag", "Hey", "Hi", "Wie geht's?",
        "Erzähl mir was über dich.", "Was kannst du?", "Was ist dein Hintergrund?",
        "Erzähl mir einen Witz.", "Wie programmiert man ein Spiel?"
    ]
    outputs = [
        "Hallo! Wie kann ich helfen?", "Guten Tag! Was kann ich für Sie tun?",
        "Hey! Wie kann ich Ihnen helfen?", "Hi! Wie kann ich behilflich sein?",
        "Mir geht's gut, danke! Wie geht es Ihnen?",
        "Ich bin ein KI-Charakter, erstellt, um mit dir zu sprechen.",
        "Ich bin darauf programmiert, hilfreiche Antworten zu geben.",
        "Mein Hintergrund ist rein virtuell.",
        "Warum ging das Buch nicht ins Kino? Es war zu spannend.",
        "Um ein Spiel zu programmieren, benötigt man eine Idee, eine Programmiersprache und viel Geduld."
    ]

    with open("training_data.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["input", "output"])
        for i in range(len(inputs)):
            writer.writerow([inputs[i], outputs[i]])

generate_training_data()

# *** Trainingsdaten laden und verarbeiten ***
try:
    data = pd.read_csv("training_data.csv")
except FileNotFoundError:
    raise Exception("Die Datei 'training_data.csv' wurde nicht gefunden. Bitte erstellen Sie sie.")

data = data.drop_duplicates()
data['language'] = data['input'].apply(lambda x: detect(x) if isinstance(x, str) else 'unknown')
data = data[data['language'] == 'de']

inputs = data['input'].values
outputs = data['output'].values

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(inputs)

input_sequences = tokenizer.texts_to_sequences(inputs)
output_sequences = tokenizer.texts_to_sequences(outputs)

max_len = max(len(seq) for seq in input_sequences)
input_padded = pad_sequences(input_sequences, maxlen=max_len, padding='post')
output_padded = pad_sequences(output_sequences, maxlen=max_len, padding='post')

X_train, X_test, y_train, y_test = train_test_split(input_padded, output_padded, test_size=0.2)

# *** KI-Modell definieren ***
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_len),
    LSTM(128, return_sequences=False),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(max_len, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training des Modells...")
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
print("Training abgeschlossen!")

def generate_ki_response(user_input):
    """Generiert eine KI-basierte Antwort."""
    seq = tokenizer.texts_to_sequences([user_input])
    padded_seq = pad_sequences(seq, maxlen=max_len, padding='post')
    prediction = model.predict(padded_seq)
    response_index = prediction.argmax(axis=-1)[0]
    response = tokenizer.sequences_to_texts([[response_index]])
    return response[0] if response else "Entschuldigung, ich habe das nicht verstanden."

# *** Flask API ***
@app.route('/chat', methods=['POST'])
@limiter.limit("100 per hour")  # Ratenbegrenzung hinzufügen
def chat():
    user_input = request.json.get('message', '').strip()
    character_name = request.json.get('character', '').strip()
    
    if not user_input:
        return jsonify({'response': "Bitte geben Sie eine Nachricht ein."})

    if character_name:
        response = character_response(character_name, user_input)
        return jsonify({'response': response})
    
    # Regelbasierte Antwort
    response = generate_ki_response(user_input)
    return jsonify({'response': response})

@app.route('/character/create', methods=['POST'])
@auth.login_required  # Nur autorisierte Benutzer können Charaktere erstellen
def create_character_api():
    data = request.json
    name = data.get('name', '').strip()
    description = data.get('description', '').strip()
    if not name or not description:
        return jsonify({'response': "Bitte geben Sie einen Namen und eine Beschreibung an."})
    response = create_character(name, description)
    return jsonify({'response': response})

@app.route('/character/select', methods=['POST'])
@auth.login_required  # Nur autorisierte Benutzer können Charaktere auswählen
def select_character_api():
    data = request.json
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'response': "Bitte geben Sie einen Charakternamen an."})
    response = select_character(name)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=False, ssl_context=('cert.pem', 'key.pem'))
