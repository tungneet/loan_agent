import asyncio
import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

client = AsyncOpenAI(api_key=st.secrets["OPENAI_API_KEY"])

SAMPLE_RATE = 16000
THRESHOLD = 1000
SILENCE_DURATION = 1.5

# Record audio until silence
def record_until_silence():
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

    return np.concatenate(recording, axis=0)

# Save to WAV file
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

# Generate reply
async def generate_reply(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly Hinglish customer support agent."},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()

# Speak
async def speak(text):
    async with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=text,
        instructions="Speak in Indian accent in Hinglish. Use natural, friendly tone.",
        response_format="pcm"
    ) as response:
        await LocalAudioPlayer().play(response)

# Streamlit UI
def main():
    st.set_page_config(page_title="🗣️ Hinglish Voice Chatbot", layout="centered")
    st.title("🗣️ Hinglish Voice Chatbot")
    st.write("Click the button and start talking. It will listen, respond, and speak back in Hinglish.")

    if st.button("🎤 Start Talking"):
        with st.spinner("Recording... Speak now!"):
            audio = record_until_silence()
            wav_path = save_to_wav(audio)

        async def process_audio():
            try:
                user_input = await transcribe(wav_path)
                st.success(f"You said: {user_input}")

                reply = await generate_reply(user_input)
                st.info(f"🤖 Bot: {reply}")

                await speak(reply)
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                os.remove(wav_path)

        asyncio.run(process_audio())

if __name__ == "__main__":
    main()
