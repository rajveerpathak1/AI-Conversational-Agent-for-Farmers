import torch
import whisper


# =========================================================
# CONFIG
# =========================================================

MODEL_SIZE = "base"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGE = "hi"  
# Use:
# "hi" for Hindi
# "en" for English
# None for auto-detect


# =========================================================
# LOAD MODEL
# =========================================================

print(f"\nLoading Whisper model: {MODEL_SIZE}")

model = whisper.load_model(MODEL_SIZE).to(DEVICE)

print(f"Using device: {DEVICE}")


# =========================================================
# TRANSCRIBE FUNCTION
# =========================================================

def transcribe(audio_path: str):

    try:

        result = model.transcribe(
            audio_path,

            language=LANGUAGE,

            fp16=torch.cuda.is_available(),

            verbose=False
        )

        return {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", [])
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    audio_file = "sample.wav"

    response = transcribe(audio_file)

    if "error" in response:

        print("\nError:")
        print(response["error"])

    else:

        print("\nTranscription:")
        print(response["text"])

        print("\nDetected Language:")
        print(response["language"])
