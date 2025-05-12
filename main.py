import asyncio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

client = AsyncOpenAI(api_key="add_key_here")

# Audio settings
SAMPLE_RATE = 16000
THRESHOLD = 1000  # adjust as needed
SILENCE_DURATION = 1.5  # seconds of silence to stop recording

# Detects silence to stop recording
def record_until_silence():
    print("🎤 Speak now...")

    recording = []
    silence_count = 0
    block_size = 1024

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16') as stream:
        while True:
            block, _ = stream.read(block_size)
            volume = np.linalg.norm(block)
            recording.append(block)

            if volume < THRESHOLD:
                silence_count += block_size / SAMPLE_RATE
                if silence_count >= SILENCE_DURATION:
                    break
            else:
                silence_count = 0

    audio = np.concatenate(recording, axis=0)
    print("🔇 Recording stopped.")
    return audio


# Save numpy audio to temp WAV file
def save_to_wav(audio_data):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wavfile.write(temp_file.name, SAMPLE_RATE, audio_data)
    return temp_file.name


# Transcribe using Whisper
async def transcribe(file_path):
    with open(file_path, "rb") as f:
        transcript = await client.audio.transcriptions.create(
            file=f,
            model="whisper-1",
            response_format="text"
        )
    return transcript.strip()


# Generate Hinglish reply using GPT-4o-mini
async def generate_reply(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()


# Speak with OpenAI TTS
async def speak(text):
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",  # Or "echo" for Indian-style voice
        input=text,
        instructions="Speak in Indian accent in Hinglish. Use natural, friendly tone.",
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)


# Main chatbot loop
async def chatbot_loop():
    while True:
        audio = record_until_silence()
        wav_path = save_to_wav(audio)

        try:
            user_input = await transcribe(wav_path)
            print(f"🗣️ You said: {user_input}")

            reply = await generate_reply(user_input)
            print(f"🤖 Bot: {reply}")

            await speak(reply)
        except Exception as e:
            print("❌ Error:", e)
        finally:
            os.remove(wav_path)


if __name__ == "__main__":
    asyncio.run(chatbot_loop())
