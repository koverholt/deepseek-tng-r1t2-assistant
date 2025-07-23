import os
import streamlit as st
from cleantext import clean
from google import genai
from google.cloud import texttospeech
from google.genai import types
from openai import OpenAI


# TTS helper function
def synthesize_speech(text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Chirp3-HD-Achernar",
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open("output/output.mp3", "wb") as out:
        out.write(response.audio_content)


# Streamlit settings
st.set_page_config(layout="wide")
st.title("DeepSeek-TNG R1T2 Assistant")

# OpenRouter client settings
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Initialize model
if "model" not in st.session_state:
    st.session_state["model"] = "tngtech/deepseek-r1t2-chimera:free"

# Initialize messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful audience-facing AI assistant. Keep your tone informal and fun. Act as the user's live sidekick for their technical talk, keep them focused, motivated, and relaxed.",
        }
    ]

# Render all chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display STT input widget
audio_value = st.audio_input("Record a voice message")

# Transcribe STT audio if used
prompt_stt = None
if audio_value:
    gemini_client = genai.Client()

    prompt_stt = gemini_client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            "Transcribe only the audio from this clip",
            types.Part.from_bytes(
                data=audio_value.read(),
                mime_type="audio/wav",
            ),
        ],
    ).text

# Main chat logic
prompt_text = st.chat_input("How can I help?")

if prompt := prompt_text or prompt_stt:

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)

        synthesize_speech(clean(response, no_emoji=True))
        st.audio("output/output.mp3", autoplay=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
