import sounddevice as sd
import numpy as np
import librosa
import joblib
import os

# --- 1. Set Audio Parameters ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 2 # Duration of live recording
MODELS_DIR = "models"

# --- 2. Load Trained Models ---
models = {}
try:
    for model_file in os.listdir(MODELS_DIR):
        if model_file.endswith(".pkl"):
            word = model_file.split(".")[0]
            model_path = os.path.join(MODELS_DIR, model_file)
            models[word] = joblib.load(model_path)
    print(f"Loaded models for: {list(models.keys())}")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run the 2_train_models.py script first.")
    exit()

# --- 3. Feature Extraction Function (Must be identical to training) ---
def extract_features(audio_signal):
    try:
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=SAMPLE_RATE, n_mfcc=13)
        delta1 = librosa.feature.delta(mfccs, order=1)
        delta2 = librosa.feature.delta(mfccs, order=2)
        features = np.concatenate((mfccs, delta1, delta2), axis=0)
        return features.T
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# --- 4. Main Recognition Loop ---
print("\n--- Live Voice Recognition ---")
print("Press Enter to record and predict, or 'q' to quit.")

while True:
    user_input = input()
    if user_input.lower() == 'q':
        break

    print("🎙️ Recording...")
    myrecording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), 
                        samplerate=SAMPLE_RATE, 
                        channels=1, 
                        dtype='float32')
    sd.wait()
    print("Recording finished. Analyzing...")

    audio_signal = myrecording.flatten()
    
    # Trim silence (optional, but good for consistency)
    audio_signal, _ = librosa.effects.trim(audio_signal)
    
    if len(audio_signal) == 0:
        print("No audio detected.")
        continue
    
    # Extract features
    test_features = extract_features(audio_signal)
    
    if test_features is None:
        print("Could not extract features.")
        continue

    # Score against all models
    scores = {}
    for word, model in models.items():
        try:
            scores[word] = model.score(test_features)
        except:
            scores[word] = -np.inf
            
    # Predict the word with the highest log-likelihood score
    predicted_word = max(scores, key=scores.get)
    
    print(f"\nPredicted Word: {predicted_word}")
    print("Scores:", scores)
    print("\nPress Enter to record again, or 'q' to quit.")

print("Exiting.")
