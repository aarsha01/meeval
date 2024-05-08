import streamlit as st
from keras.models import load_model
from music21 import stream, note, chord
import numpy as np
import os
import json
from pydub import AudioSegment
import tempfile
from midi2audio import FluidSynth
# Load data from static JSON file
with open('static/data.json') as f:
    data = json.load(f)

Note_Count = 250
length = data['length']
L_symb = data['L_symb']
reverse_mapping = data['reverse_mapping']
X_seed = np.array(data['X_seed'])  # Convert list back to numpy array

# Load the LSTM model
model = load_model('LSTM_MODEL_MAIN.h5')

# Function to convert a list of chords and notes into a music stream
def chords_n_notes(snippet):
    melody = []
    offset = 0  # Incremental

    for i in snippet:
        # If it is a chord
        if ("." in i or i.isdigit()):
            chord_notes = i.split(".")  # Separating the notes in a chord
            notes = []

            for j in chord_notes:
                # Check if the pitch name is valid
                if j in ('C', 'D', 'E', 'F', 'G', 'A', 'B'):
                    inst_note = note.Note(j)
                    notes.append(inst_note)
                else:
                    # Handle invalid pitch name (replace with default pitch)
                    print(f"Invalid pitch name: {j}. Using default pitch.")
                    inst_note = note.Note('C')  # Default to 'C'
                    notes.append(inst_note)

            chord_snip = chord.Chord(notes)
            chord_snip.offset = offset
            melody.append(chord_snip)
        # If it is a note
        else:
            # Check if the pitch name is valid
            if i in ('C', 'D', 'E', 'F', 'G', 'A', 'B'):
                note_snip = note.Note(i)
            else:
                # Handle invalid pitch name (replace with default pitch)
                print(f"Invalid pitch name: {i}. Using default pitch.")
                note_snip = note.Note('C')  # Default to 'C'
            note_snip.offset = offset
            melody.append(note_snip)

        # Increase offset each iteration so that notes do not stack
        offset += 1

    melody_stream = stream.Stream(melody)
    return melody_stream


# Function to generate melody
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
        Notes_Generated.append(index)

        # Map indices to their corresponding note or chord symbols
        Music = [reverse_mapping.get(char, 'Unknown') for char in Notes_Generated]

        seed = np.insert(seed[0], len(seed[0]), index_N)
        seed = seed[1:]

    # Now, we have music in the form of a list of chords and notes, and we want to create a MIDI file.
    Melody = chords_n_notes(Music)
    Melody_midi = stream.Stream(Melody)

    # Save the melody to a temporary MIDI file
    midi_temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
    midi_temp_file.close()
    Melody_midi.write('midi', midi_temp_file.name)

    # Convert the MIDI file to MP3 audio using pydub
    # Convert the MIDI file to WAV using midi2audio
    wav_temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wav_temp_file.close()

    FluidSynth().midi_to_audio(midi_temp_file.name, wav_temp_file.name)

    # Convert the WAV file to MP3 using pydub
    audio = AudioSegment.from_wav(wav_temp_file.name)
    mp3_temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
    mp3_temp_file.close()
    audio.export(mp3_temp_file.name, format='mp3')

    return mp3_temp_file.name

# Define a function to let the user dynamically select a musician
def select_musician():
    musicians = ["chopin", "bach", "beeth", "haydn", "liszt", "chopin_beeth"]
    selected_musician = st.selectbox("Select a musician:", musicians)
    return selected_musician

# Streamlit app
st.title('Music Generator')

selected_musician = select_musician()

if st.button('Generate Music'):
    # Add logic to handle selected musician if needed
    mp3_file_path = Melody_Generator(Note_Count, model, length, L_symb, reverse_mapping)

    # Stream the generated MP3 audio file
    st.audio(mp3_file_path, format='audio/mp3')
