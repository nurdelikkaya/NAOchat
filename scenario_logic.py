#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scenario_logic.py

Python 3 Flask server that:
 - /listenUser -> receives an audio file from Pepper for STT (Hugging Face),
   optionally calls ChatGPT for a "novel" response, then does TTS with Google
   Cloud (Turkish) and returns WAV audio as base64 + recognized text.
 - /ttsBytes -> text-to-speech for any prompt or scenario lines, returns base64 WAV
 - /startScenario -> resets scenario state, if needed

Usage:
  python3 scenario_logic.py
"""

import os
import base64
import uuid
import json
from pydantic import BaseModel
import openai
from openai import OpenAI
import requests
from flask import Flask, request, jsonify
from google.cloud import texttospeech
from google.cloud import speech


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "tts_credentials"
)

os.environ["GOOGLE_STT_CREDENTIALS"] = os.getenv(
    "GOOGLE_STT_CREDENTIALS",
    "stt_credentials"
)

app = Flask(__name__)

tts_client = texttospeech.TextToSpeechClient()


client = OpenAI(
    api_key=("api_key"),  
)

session = {
    'chat_history': []
}


# If you have scenario lines, you can store them in a global list or DB
SCENARIO_LINES = [
    "Merhaba! Ben bir NAO robotuyum. Hazır mısın?",
    "Kalem nesnesi: 3 dakikanız var, aklınıza gelen yaratıcı fikirleri söyleyin!",
    "Zaman doldu, şimdi plastik şişeye geçiyoruz.",
    "Plastik şişe nesnesi: 3 dakikanız var, neler yapılabilir?",
    "Zaman doldu, deney bitti, teşekkürler!"
]

# ------------------------------------------------------------------------------
# HELPER: Google TTS
# ------------------------------------------------------------------------------
def google_tts_turkish(text):
    """
    Generate TTS using Google Cloud API with specified voice and pitch.

    Parameters:
        text (str): Text to synthesize.

    Returns:
        bytes: The raw WAV audio content.
    """
    if not isinstance(text, str) or not text.strip():
        print(f"[google_tts_turkish] Invalid text input: {text}")
        return None

    try:
        # Define the input text
        synthesis_input = texttospeech.SynthesisInput(text=text)
        
        # Define the voice parameters
        voice = texttospeech.VoiceSelectionParams(
            language_code="tr-TR",      # Turkish language
            name="tr-TR-Standard-D",   # Specific voice name
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        
        # Define the audio configuration
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,  # WAV format
            pitch=-4.0,                                         # Lower pitch by 4 semitones
            speaking_rate=1.0                                   # Normal speaking speed
        )
        
        # Call the TTS API
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content  # Return raw WAV bytes

    except Exception as e:
        print("[google_tts_turkish] Error:", e)
        return None
# ------------------------------------------------------------------------------
# HELPER: Google STT
# ------------------------------------------------------------------------------
def google_stt(wav_data):
    """
    Perform STT using Google Cloud Speech-to-Text.
    """
    if not wav_data or len(wav_data) == 0:
        print("[google_stt] Empty or invalid WAV data.")
        return ""

    try:
        client = speech.SpeechClient()

        # Configure audio settings
        audio = speech.RecognitionAudio(content=wav_data)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="tr-TR"  # Turkish language
        )

        # Perform speech recognition
        response = client.recognize(config=config, audio=audio)

        # Extract transcription
        for result in response.results:
            return result.alternatives[0].transcript  # Return the first transcript

        return ""  # No transcription found
    except Exception as e:
        print("[google_stt] Exception:", e)
        return ""

# ------------------------------------------------------------------------------
# HELPER: ChatGPT
# ------------------------------------------------------------------------------
def chatgpt_respond(prompt_text):
    """
    Generate a creative response using ChatGPT.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Sen Deniz adlı bir NAO insansı robotsun. Katılımcılarla tamamen Türkçe olarak etkileşim kuruyor ve onlara belirtilen bir gündelik nesnenin yaratıcı alternatif kullanımları için fikirler üretmelerine yardımcı oluyorsun. Tanışma faslını bitirdik ve merhabalaştınız. Katılımcıya görevi açıkladık ve nesne için toplamda 3 dakika konuşacağınızı belirttik. Görevin, katılımcıya rehberlik ederek sorular sormak, fikirlerini geliştirmelerine destek olmak ve yaratıcı öneriler sunmaktır. Cevaplarını doğal bir diyalog sürdürebilmek için olabildiğince kısa tut ve doğal bir dil kullan. Öneri vermeye kullanıcı başlayacak, daha sonra sen başka bir öneri sun, sonrasında kullanıcıya başka nasıl kullanılabileceğini sor, böylece bir sen bir kullanıcı bir kullanım önersin. Süre doldu promptu gelene kadar aynı nesne üzerinde duracağız, bu nedenle kullanıcı takılırsa da yapıcı bir şekilde yardımcı ol, böylece belli bir cevaba erişmesini sağla. Farklı bir nesne üzerine düşünmeyi önerme."
                )},
                *session["chat_history"],
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7
        )
        response = completion.choices[0].message.content.strip()  # Extract text content
        session["chat_history"].append({"role": "assistant", "content": response})
        return response
    except Exception as e:
        print("[chatgpt_respond] Error:", e)
        return "Bir hata oluştu."

    
# ------------------------------------------------------------------------------
# ROUTES
# ------------------------------------------------------------------------------

@app.route("/startScenario", methods=["GET"])
def start_scenario():
    """
    Resets scenario or sets flags, if needed.
    """
    # Reset scenario state
    session['chat_history'] = []
    return jsonify({"message": "Scenario started."})

@app.route("/ttsBytes", methods=["GET"])
def tts_bytes():
    """
    TTS any text. Return JSON with base64 WAV data: { "wav_base64": "..." }
    Usage: /ttsBytes?prompt=Merhaba%20Dunya
    """
    prompt = request.args.get("prompt")
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    wav_data = google_tts_turkish(prompt)
    if wav_data is None:
        return jsonify({"error": "TTS failed"}), 500

    b64_data = base64.b64encode(wav_data).decode("utf-8")
    return jsonify({"wav_base64": b64_data})

@app.route("/listenUser", methods=["POST"])
def listen_user():
    """
    Process user audio input:
    - Perform STT
    - Generate a ChatGPT response
    - Convert ChatGPT response to TTS
    - Return JSON with recognized text, ChatGPT response, and TTS audio
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file_ = request.files["file"]
    wav_data = file_.read()

    # Get current instruction from the request
    current_instruction = request.form.get("current_instruction", "")

    # Speech-to-Text (STT)
    recognized_text = google_stt(wav_data)
    if not recognized_text:
        return jsonify({"error": "STT failed", "recognized_text": ""}), 500

    # Generate ChatGPT response including the current instruction
    prompt_text = f"{current_instruction}\nKullanıcı: {recognized_text}"
    chatgpt_res = chatgpt_respond(prompt_text)
    if not chatgpt_res:
        return jsonify({"error": "ChatGPT failed", "recognized_text": recognized_text}), 500

    # Convert ChatGPT response to TTS
    audio_bytes = google_tts_turkish(chatgpt_res)
    if audio_bytes is None:
        return jsonify({"error": "TTS failed", "recognized_text": recognized_text, "chatgpt_response": chatgpt_res}), 500

    # Return JSON response
    b64_data = base64.b64encode(audio_bytes).decode("utf-8")
    return jsonify({
        "recognized_text": recognized_text,
        "chatgpt_response": chatgpt_res,
        "wav_base64": b64_data
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
