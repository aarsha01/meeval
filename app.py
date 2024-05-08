from flask import Flask, render_template, request, json, jsonify
from tensorflow.keras.models import load_model
from music21 import stream, note, chord
import numpy as np
import os,base64
from io import BytesIO

app = Flask(__name__)



# Load the LSTM model and other necessary data
lstm_model = load_model('models/LSTM_MODEL_MAIN.h5') 

with open('static/data.json') as f:
    data = json.load(f)

Note_Count = 250
length = data['length']
L_symb = data['L_symb']
reverse_mapping = data['reverse_mapping']
X_seed = np.array(data['X_seed'])  # Convert list back to numpy array

def chords_n_notes(snippet):
    melody = []
    offset = 0  # Incremental

    for item in snippet:
        if isinstance(item, chord.Chord):  # Check if it's a chord object
            chord_snip = item
            chord_snip.offset = offset
            melody.append(chord_snip)
        elif isinstance(item, note.Note):  # Check if it's a note object
            note_snip = item
            note_snip.offset = offset
            melody.append(note_snip)

        # Increase offset each iteration so that notes do not stack
        offset += 1

    melody_stream = stream.Stream(melody)
    return melody_stream


def Melody_Generator(Note_Count, model, length, L_symb, reverse_mapping):
    seed = X_seed[np.random.randint(0, len(X_seed) - 1)]
    Music = []
    Notes_Generated = []

    for i in range(Note_Count):
        seed = seed.reshape(1, length, 1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction) / 1.0  # diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index / float(L_symb)
      
        
        # Convert index_N to an integer or chord representation
        if index_N.is_integer():
            Notes_Generated.append(int(index_N))
        else:
            index_N_int = int(index_N)  # Integer part of index_N
            if str(index_N_int) in reverse_mapping:
                # Check if the integer part is a valid key in reverse_mapping
                chord_notes = reverse_mapping[str(index_N_int)].split(".")
                Notes_Generated.extend([int(note) for note in chord_notes])
            else:
                # Handle the case when the integer part is not a key in reverse_mapping
                print("Chord index not found in reverse_mapping:", index_N_int)

        # Map indices to their corresponding note or chord symbols, handling KeyError
        
        Music = [reverse_mapping.get(str(char), 'Unknown') for char in Notes_Generated]

        seed = np.insert(seed[0], len(seed[0]), index_N)
        seed = seed[1:]

    # Convert the generated music to a music stream
    Melody = chords_n_notes(Music)
    
    return Music, Melody

@app.route('/')
def index():
    # Use the loaded values as needed in your route function
    return render_template('firstpage.html', Note_Count=Note_Count, length=length, L_symb=L_symb, reverse_mapping=reverse_mapping, X_seed=X_seed)

@app.route('/generate_music', methods=['POST'])
def generate_music_endpoint():
    print("Received request data:", request.json)
    musician = request.json['musician']
    generated_music, music_stream = Melody_Generator(Note_Count, lstm_model, length, L_symb, reverse_mapping)
    print("music_stream", music_stream)

    # Write the MIDI data to a temporary file
    temp_file = "temp_music.mid"
    music_stream.write('midi', fp=temp_file)

    # Read the content of the temporary file and encode it to base64
    with open(temp_file, "rb") as f:
        base64_music = base64.b64encode(f.read()).decode('utf-8')

    # Delete the temporary file
    os.remove(temp_file)

    return jsonify({'generated_music': base64_music})



if __name__ == '__main__':
    app.run(debug=True,port =5000)