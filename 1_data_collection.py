import sounddevice as sd
import soundfile as sf
import os

# --- 1. Set Audio Parameters ---
SAMPLE_RATE = 16000
RECORD_SECONDS = 2  # Keep recordings short and focused
DATA_DIR = "data"   # Directory to store our recordings

# --- 2. Define Words to Record ---
WORDS = ["start", "stop", "light_on", "light_off"]
NUM_SAMPLES = 20  # Number of samples to record for each word

def record_and_save():
    """Records audio and saves it to the correct directory."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for word in WORDS:
        word_dir = os.path.join(DATA_DIR, word)
        if not os.path.exists(word_dir):
            os.makedirs(word_dir)
        
        print(f"\n--- Recording for word: '{word}' ---")
        print(f"Press Enter to start recording {NUM_SAMPLES} samples.")
        input() # Wait for user to press Enter

        for i in range(NUM_SAMPLES):
            print(f"Recording sample {i+1}/{NUM_SAMPLES} for '{word}'. Get ready...")
            
            # 3-second countdown
            for j in range(3, 0, -1):
                print(f"{j}...", end="", flush=True)
                sd.sleep(1000)
            
            print("SPEAK!")
            
            myrecording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), 
                                samplerate=SAMPLE_RATE, 
                                channels=1, 
                                dtype='float32')
            sd.wait()
            print("Done.")

            filename = os.path.join(word_dir, f"{i+1}.wav")
            sf.write(filename, myrecording, SAMPLE_RATE)

    print("\n--- Data collection complete. ---")

if __name__ == "__main__":
    record_and_save()
