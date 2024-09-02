# prompt: Make a streamlit app to do the same , also convert the first 30 seconds of MP3 to wav before doing it and save the file as app.py 

import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import sys
sys.path.append("/workspaces/Music-Genre-Classifier/models/research/audioset/vggish")
import vggish_input
import vggish_input
import vggish_params
import vggish_slim
import soundfile as sf
import pydub
from pydub import AudioSegment

# Load the trained model
model = load_model('/workspaces/Music-Genre-Classifier/nmodel86.h5')  # Replace with the actual path to your model

# Define the genre labels
genre_labels = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Function to extract VGGish embeddings
def extract_embeddings(audio_file):
  with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, '')  # Replace with the actual path to the VGGish checkpoint
    features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

    examples_batch = vggish_input.wavfile_to_examples(audio_file)
    [embedding] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})

  return np.mean(embedding, axis=0)

# Function to predict the genre
def predict_genre(audio_file):
  embedding = extract_embeddings(audio_file)
  prediction = model.predict(np.expand_dims(embedding, axis=0))
  predicted_class = np.argmax(prediction)
  return genre_labels[predicted_class]

# Streamlit app
def main():
  st.title("Music Genre Classifier")
  st.write("Upload an audio file (MP3 or WAV) to classify its genre.")

  uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

  if uploaded_file is not None:
    # Convert MP3 to WAV if necessary
    if uploaded_file.name.endswith(".mp3"):
      audio = AudioSegment.from_mp3(uploaded_file)
      # Extract first 30 seconds
      audio = audio[:30000]  # 30 seconds in milliseconds
      wav_file = "temp.wav"
      audio.export(wav_file, format="wav")
      audio_path = wav_file
    else:
      audio_path = "temp.wav"  # Save the uploaded WAV file temporarily
      with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the genre
    try:
      predicted_genre = predict_genre(audio_path)
      st.write(f"Predicted Genre: **{predicted_genre}**")
    except Exception as e:
      st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
  main()
