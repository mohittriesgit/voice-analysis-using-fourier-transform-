import numpy as np
import librosa
import os
import glob
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
import joblib # For saving models

DATA_DIR = "data"
MODELS_DIR = "models"
SAMPLE_RATE = 16000
N_COMPONENTS = 5 # Number of HMM states
N_MIX = 3        # Number of Gaussian mixtures per state
ITERATIONS = 200

# --- Step 1: Feature Extraction Function (MFCCs + Deltas) ---
def extract_features(audio_path):
    """
    Extracts 39 features (13 MFCCs + 13 deltas + 13 delta-deltas)
    from an audio file.
    """
    try:
        audio_signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Trim silence from the beginning and end
        audio_signal, _ = librosa.effects.trim(audio_signal)
        
        if len(audio_signal) == 0:
            return None # Skip empty files

        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=13)
        delta1 = librosa.feature.delta(mfccs, order=1)
        delta2 = librosa.feature.delta(mfccs, order=2)

        features = np.concatenate((mfccs, delta1, delta2), axis=0)
        
        # Transpose to get (num_frames x 39)
        return features.T
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# --- Step 2: HMM Training Function ---
def train_hmm_model(features_list):
    """
    Trains a GMMHMM model.
    `features_list` is a list of feature matrices (one for each audio file).
    """
    # Concatenate all features into one big matrix
    all_features = np.concatenate(features_list)
    
    # Get the lengths of each individual feature matrix
    lengths = [f.shape[0] for f in features_list]
    
    # --- Create a Left-to-Right HMM ---
    model = hmm.GMMHMM(n_components=N_COMPONENTS, n_mix=N_MIX, n_iter=ITERATIONS, 
                         covariance_type="diag", init_params="cm", params="cmt")
    
    # Set up a left-to-right transition probability matrix
    transmat = np.zeros((N_COMPONENTS, N_COMPONENTS))
    for i in range(N_COMPONENTS):
        if i < N_COMPONENTS - 1:
            transmat[i, i] = 0.5
            transmat[i, i + 1] = 0.5
        else:
            transmat[i, i] = 1.0 # Last state loops
    
    model.transmat_ = transmat
    
    # Set start probability (always start in state 0)
    startprob = np.zeros(N_COMPONENTS)
    startprob[0] = 1.0
    model.startprob_ = startprob
    
    # --- Train the model ---
    model.fit(all_features, lengths=lengths)
    
    return model

# --- Main Training and Testing ---
if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    words = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    all_test_files = {}
    
    for word in words:
        print(f"\n--- Processing word: {word} ---")
        word_dir = os.path.join(DATA_DIR, word)
        
        # Load all .wav files
        all_files = glob.glob(os.path.join(word_dir, "*.wav"))
        
        # Split data: 80% for training, 20% for testing
        train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
        all_test_files[word] = test_files # Save test files for later
        
        # --- Extract features for training files ---
        train_features = []
        for file_path in train_files:
            features = extract_features(file_path)
            if features is not None:
                train_features.append(features)
        
        if not train_features:
            print(f"No valid training features for {word}. Skipping.")
            continue
            
        # --- Train the model ---
        print(f"Training HMM for '{word}' with {len(train_features)} samples...")
        hmm_model = train_hmm_model(train_features)
        
        # --- Save the model ---
        model_path = os.path.join(MODELS_DIR, f"{word}.pkl")
        joblib.dump(hmm_model, model_path)
        print(f"Model for '{word}' saved to {model_path}")

    # --- Step 3: Test the models ---
    print("\n--- Testing Models ---")
    correct_predictions = 0
    total_samples = 0
    
    # Load all trained models
    trained_models = {}
    for word in words:
        model_path = os.path.join(MODELS_DIR, f"{word}.pkl")
        if os.path.exists(model_path):
            trained_models[word] = joblib.load(model_path)
            
    for true_word, test_files in all_test_files.items():
        for test_file in test_files:
            total_samples += 1
            test_features = extract_features(test_file)
            
            if test_features is None:
                continue
                
            scores = {}
            for word, model in trained_models.items():
                try:
                    scores[word] = model.score(test_features)
                except:
                    scores[word] = -np.inf # Error scoring (e.g., empty features)
            
            # Predict the word with the highest log-likelihood score
            predicted_word = max(scores, key=scores.get)
            
            if predicted_word == true_word:
                correct_predictions += 1
            
            print(f"File: {test_file} | True: {true_word} | Predicted: {predicted_word} (Score: {scores[predicted_word]:.2f})")

    accuracy = (correct_predictions / total_samples) * 100
    print("\n--- Test Results ---")
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
