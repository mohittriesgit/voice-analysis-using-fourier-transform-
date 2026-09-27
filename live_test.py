import sys
import fake_rpi
from fake_rpi import RPi
sys.modules['RPi'] = RPi
import sounddevice as sd
import numpy as np
import librosa
import joblib
import os
import time
try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    # If RPi.GPIO fails (because we are on Windows), use the fake one
    import fake_rpi
    from fake_rpi import RPi
    GPIO = RPi.GPIO
    print("⚠️  Running on Windows: Using Fake RPi.GPIO")

# --- 1. Hardware Setup ---
RELAY_PIN = 17 # The GPIO pin connected to the relay's IN pin

# Set up the GPIO pins
GPIO.setmode(GPIO.BCM) # Use Broadcom pin-numbering scheme
GPIO.setup(RELAY_PIN, GPIO.OUT, initial=GPIO.HIGH) # Set pin as output, start as OFF (High is OFF for most relays)

# --- 2. Configuration & Model Loading ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 2 # Duration of each listening window
MODELS_DIR = "models"

print("Loading models...")
models = {}
try:
    # Load all .pkl files from the models directory
    for model_file in os.listdir(MODELS_DIR):
        if model_file.endswith(".pkl"):
            word = model_file.split(".")[0]
            model_path = os.path.join(MODELS_DIR, model_file)
            models[word] = joblib.load(model_path)
            print(f" - Loaded model for: '{word}'")
    
    if not models:
        print("❌ No models found in the 'models' directory.")
        exit()
        
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Please make sure you have transferred the 'models' folder to the Pi.")
    exit()

print("Models loaded successfully.")

# --- 3. Feature Extraction Function ---
# (This must be EXACTLY the same as used during training)
def extract_features(audio_signal):
    try:
        # Extract 13 MFCCs
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=SAMPLE_RATE, n_mfcc=13)
        # Calculate Delta (1st derivative)
        delta1 = librosa.feature.delta(mfccs, order=1)
        # Calculate Delta-Delta (2nd derivative)
        delta2 = librosa.feature.delta(mfccs, order=2)
        # Combine into a single 39-dimensional feature vector per frame
        features = np.concatenate((mfccs, delta1, delta2), axis=0)
        # Transpose to get shape (n_frames, 39)
        return features.T
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# --- 4. Hardware Control Function ---
def execute_command(command):
    """Triggers the relay based on the recognized command."""
    if command == "light_on":
        print("💡 ACTION: Turning Light ON")
        # Set pin LOW to turn the relay ON (for active-low relays)
        GPIO.output(RELAY_PIN, GPIO.LOW)
    
    elif command == "light_off":
        print("🌑 ACTION: Turning Light OFF")
        # Set pin HIGH to turn the relay OFF
        GPIO.output(RELAY_PIN, GPIO.HIGH)
    
    else:
        print(f"Unknown command: {command}. No action taken.")

# --- 5. Main Recognition Loop ---
print("\n---------------------------------------")
print("🎙️  Voice Command System is READY.")
print("Speaking 'light on' or 'light off'.")
print("Press Ctrl+C to exit.")
print("---------------------------------------\n")

try:
    while True:
        print("Listening...", end="\r", flush=True)
        
        # 1. Record Audio
        myrecording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), 
                            samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait() # Wait for recording to finish
        
        # 2. Preprocess Audio
        audio_signal = myrecording.flatten()
        # Trim silence to focus on the speech content
        audio_signal, _ = librosa.effects.trim(audio_signal, top_db=20)
        
        # If audio is too short after trimming (just noise), skip it
        if len(audio_signal) < (SAMPLE_RATE * 0.5): 
            continue

        print("Analyzing...   ", end="\r", flush=True)

        # 3. Extract Features
        test_features = extract_features(audio_signal)
        
        if test_features is not None:
            # 4. Score against all models
            scores = {}
            for word, model in models.items():
                try:
                    scores[word] = model.score(test_features)
                except:
                    scores[word] = -np.inf # Assign very low score on error
            
            # Find the model with the highest score
            predicted_word = max(scores, key=scores.get)
            best_score = scores[predicted_word]

            # Optional: Add a confidence threshold
            # If the score is too low, it's likely just noise.
            # You may need to adjust this value based on your testing.
            CONFIDENCE_THRESHOLD = -5000 
            
            if best_score > CONFIDENCE_THRESHOLD:
                print(f"\n🗣️  Recognized: '{predicted_word}' (Score: {best_score:.0f})")
                # 5. Execute the physical action
                execute_command(predicted_word)
            else:
                print(f"\n🤷 Not recognized. (Best score: {best_score:.0f} was too low)")

        # Small pause before next listening cycle
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n\nStopping program by user request.")

finally:
    # Clean up GPIO pins properly so they aren't left in an active state
    print("Cleaning up GPIO pins...")
    GPIO.cleanup()
    print("System shut down.")
