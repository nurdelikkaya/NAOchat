#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
pepper_bridge.py

Python 2.7 code:
 - Connects to Pepper via naoqi
 - Records audio from Pepper's mic using ALAudioRecorder (4 mics)
 - Then sums the 4 channels into a single mono WAV
 - Sends that WAV to scenario_logic.py (/listenUser) for STT + optional ChatGPT + TTS
 - Receives base64 WAV, decodes, scps to Pepper, plays
 - Monitors user inactivity (10s) -> "Sen düşün ben beklerim"
 - If STT/LLM/TTS >5s, insert filler phrase
 - 3 min for each object
"""
import random
import sys
import os
import time
import base64
import wave
import struct
import paramiko
import subprocess
from optparse import OptionParser
from naoqi import ALBroker, ALModule, ALProxy


ROBOT_IP = "robot_ip"  
ROBOT_PORT = robot_port
NAO_PASSWORD = "nao_password"
managerProxy = ALProxy("ALBehaviorManager", ROBOT_IP, ROBOT_PORT)

SCENARIO_SERVER_HOST = "scenario_server_hsot"
SCENARIO_SERVER_PORT = scenario_server_port

LOCAL_TEMP_DIR = "local_temp_dir"
PEPPER_TEMP_DIR = "pepper_temp_dir"

FILLER_PHRASES = [u"Düşüneyim", u"Bir saniye"]

IDLE_MESSAGES = [
    u"Sen düşün, ben beklerim.",
    u"İstersen biraz daha düşünebiliriz.",
    u"Merak etme bekliyorum."
]


# -------------------------------------------------------------------------------
# Utility function to mix (sum) a multichannel WAV down to mono
# -------------------------------------------------------------------------------

def sum_to_mono(in_wav_file, out_wav_file):
    import wave, struct

    w_in = None
    w_out = None
    try:
        w_in = wave.open(in_wav_file, 'rb')
        params = w_in.getparams()  # (nchannels, sampwidth, framerate, nframes, ...)
        n_channels, sampwidth, framerate, n_frames = params[0], params[1], params[2], params[3]

        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM supported by sum_to_mono() in this example.")

        frames = w_in.readframes(n_frames)
        # Close the input file early (it's fully read)
        w_in.close()
        w_in = None

        samples = struct.unpack("<{}h".format(n_frames * n_channels), frames)

        # Average across channels
        mono_samples = []
        for i in range(0, len(samples), n_channels):
            frame_channels = samples[i : i + n_channels]
            avg = sum(frame_channels) // n_channels
            mono_samples.append(avg)

        mono_data = struct.pack("<{}h".format(len(mono_samples)), *mono_samples)

        w_out = wave.open(out_wav_file, 'wb')
        w_out.setnchannels(1)
        w_out.setsampwidth(sampwidth)
        w_out.setframerate(framerate)
        w_out.writeframes(mono_data)
        w_out.close()
        w_out = None

        print("[sum_to_mono] Successfully wrote mono WAV:", out_wav_file)

    except Exception as e:
        print("[sum_to_mono] Error mixing to mono:", e)
    finally:
        # In case of an error above, close files if still open
        if w_in:
            try:
                w_in.close()
            except:
                pass
        if w_out:
            try:
                w_out.close()
            except:
                pass

# Gestures for speaking
SPEAKING_GESTURES = [
    "animations/Stand/BodyTalk/Speaking/BodyTalk_8",
    "animations/Stand/BodyTalk/Speaking/BodyTalk_10",
    "animations/Stand/BodyTalk/Speaking/BodyTalk_1",
    "animations/Stand/BodyTalk/Speaking/BodyTalk_14",
    "animations/Stand/BodyTalk/Speaking/BodyTalk_20"
]

def wave_hand(posture_proxy, motion_proxy, hand="right", speed=2):
    """
    Waves the specified hand and moves the fingers accordingly.
    """
    try:
        right_arm_joint_names = ["LShoulderRoll", "LShoulderPitch","LElbowRoll", "LElbowYaw", "LWristYaw"]

        # Set initial positions
        initial_positions = [1, 0.0,0.0, 0.0, 0.0]
        motion_proxy.angleInterpolationWithSpeed(right_arm_joint_names, initial_positions, 0.5)
        position2 = [1.0,-1.7 ,-1.0, 0.0, 0.0]
        motion_proxy.angleInterpolationWithSpeed(right_arm_joint_names, position2, 0.5)
        position3 = [1.0,-1.7, -0.1, 0.0, 0.0]
        motion_proxy.angleInterpolationWithSpeed(right_arm_joint_names, position3, 0.5)
        position4 = [1.0,-1.7, -1.0, 0.0, 0.0]
        motion_proxy.angleInterpolationWithSpeed(right_arm_joint_names, position4, 0.5)
        print("[wave_hand] {} hand wave completed successfully.".format(hand.capitalize()))
        posture_proxy.goToPosture("Stand")
    except Exception as e:
        print("[wave_hand] Error waving {} hand: {}".format(hand, e))

def launch_random_gestures(managerProxy, gestures, duration):
    """
    Launch random gestures with an 8-second wait between them.
    If there isn't enough time left to complete another gesture, skip launching it.
    """
    start_time = time.time()
    current_gesture = None
    gesture_duration = 8.0

    try:
        while time.time() - start_time < duration:
            remaining_time = duration - (time.time() - start_time)
            if remaining_time < gesture_duration:
                print("[Speaking] Not enough time for another gesture. Stopping gestures.")
                break

            if current_gesture:
                managerProxy.stopBehavior(current_gesture)

            current_gesture = random.choice(gestures)
            print("[Speaking] Launching gesture: {}".format(current_gesture))
            managerProxy.post.runBehavior(current_gesture)

            time.sleep(gesture_duration)
    except Exception as e:
        print("[Speaking] Error managing gestures: {}".format(e))
    finally:
        if current_gesture:
            try:
                managerProxy.stopBehavior(current_gesture)
                print("[Speaking] Stopped gesture: {}".format(current_gesture))
            except Exception as e:
                print("[Speaking] Error stopping final gesture: {}".format(e))

def get_wav_duration(filepath):
    try:
        wav_file = wave.open(filepath, 'r')
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        wav_file.close()
        duration = frames / float(rate)
        return duration
    except Exception as e:
        print("[get_wav_duration] Error reading WAV file: {}".format(e))
        return 0

# -------------------------------------------------------------------------------
# Helper: Minimal HTTP POST to /listenUser
# -------------------------------------------------------------------------------
def post_audio_for_stt(audio_path, current_instruction=""):
    """
    POST the audio file to scenario_logic.py: /listenUser
    Expecting JSON with { recognized_text, chatgpt_response, wav_base64 }

    'current_instruction' is appended to the ChatGPT prompt,
    ensuring lines like "Şimdi kalem nesnesi..." are part of the conversation context.
    """
    import requests
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}
            # The 'current_instruction' will be appended to user text on the server side
            data = {"current_instruction": current_instruction}
            url = "http://{}:{}/listenUser".format(SCENARIO_SERVER_HOST, SCENARIO_SERVER_PORT)
            start_t = time.time()
            r = requests.post(url, files=files, data=data, timeout=60)
            delay = time.time() - start_t

        if r.status_code == 200:
            return r, delay
        else:
            print("[post_audio_for_stt] HTTP error:", r.status_code, r.text)
            return None, delay
    except Exception as e:
        print("[post_audio_for_stt] Exception:", e)
        return None, 0

def download_tts_to_file(prompt, local_path):
    import requests
    import base64
    
    # Build the URL with query params rather than POST
    tts_url = "http://{}:{}/ttsBytes".format(SCENARIO_SERVER_HOST, SCENARIO_SERVER_PORT)
    # If your server only recognizes ?prompt=... for text:
    params = {"prompt": prompt}

    try:
        r = requests.get(tts_url, params=params, timeout=10)
        if r.status_code != 200:
            print("[download_tts_to_file] Error:", r.text)
            return False

        js = r.json()
        if "wav_base64" not in js:
            print("[download_tts_to_file] 'wav_base64' not found in response")
            return False

        wav_data = base64.b64decode(js["wav_base64"])
        with open(local_path, "wb") as f:
            f.write(wav_data)

        return True
    except Exception as e:
        print("[download_tts_to_file] exception:", e)
        return False


def scp_and_play(local_path, remote_filename, audio_player):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(ROBOT_IP, username="nao", password=NAO_PASSWORD)
        sftp = ssh.open_sftp()
        remote_path = os.path.join(PEPPER_TEMP_DIR, remote_filename)
        sftp.put(local_path, remote_path)
        sftp.close()
        ssh.close()
        audio_player.playFile(remote_path)
    except Exception as e:
        print("[scp_and_play] error:", e)

# -------------------------------------------------------------------------------
# Module
# -------------------------------------------------------------------------------
class PepperBridge(ALModule):
    def __init__(self, name, pip, pport):
        ALModule.__init__(self, name)
        self.audio_recorder = None
        self.audio_player = None

        print("[PepperBridge] Connecting to Pepper proxies...")
        try:
            self.audio_recorder = ALProxy("ALAudioRecorder", pip, pport)
            self.audio_player = ALProxy("ALAudioPlayer", pip, pport)
        except Exception as e:
            print("[PepperBridge] connection error:", e)
            sys.exit(1)
        print("[PepperBridge] Connected to Pepper audio modules.")

    def record_audio(self, filepath, duration=3):
        """
        Record from Pepper's mic for 'duration' seconds (4-ch),
        SCP to local 'filepath', then mix to mono.
        """
        remote_pepper_record_path = "/home/nao/recordings/capture.wav"
        # Record from all 4 mics
        channels = [1, 1, 1, 1]
        sampleRate = 16000

        try:
            self.audio_recorder.stopMicrophonesRecording()
        except RuntimeError:
            pass  # Not recording currently

        try:
            # 1) Record multi-channel WAV on Pepper
            self.audio_recorder.startMicrophonesRecording(
                remote_pepper_record_path, "wav", sampleRate, channels
            )
            time.sleep(duration)
            self.audio_recorder.stopMicrophonesRecording()

            # 2) SCP from Pepper to local
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(ROBOT_IP, username="nao", password=NAO_PASSWORD)
            sftp = ssh.open_sftp()
            sftp.get(remote_pepper_record_path, filepath)
            sftp.close()
            ssh.close()

            # 3) Convert to single-channel
            sum_to_mono(filepath, filepath)

        except Exception as e:
            print("[record_audio] error:", e)

# ------------------------------------------------------------------------------
# Behavior management
# ------------------------------------------------------------------------------
def launchAndStopBehavior(managerProxy, behaviorName):
    if managerProxy.isBehaviorInstalled(behaviorName):
        if not managerProxy.isBehaviorRunning(behaviorName):
            managerProxy.post.runBehavior(behaviorName)
        else:
            print("Behavior is already running.")
    else:
        print("Behavior not found.")
        return

def stopBehavior(managerProxy, behaviorName):
    if managerProxy.isBehaviorRunning(behaviorName):
        managerProxy.stopBehavior(behaviorName)
    else:
        print("Behavior is already stopped.")

def start_face_tracking(tracker, face_detection):
    try:
        face_detection.subscribe("FaceTracking")
        tracker.registerTarget("Face", 0.1)
        tracker.setMode("Head")
        tracker.track("Face")
        print("[FaceTracking] Face tracking started.")
    except Exception as e:
        print("[FaceTracking] Error starting face tracking:", e)



def main():
    parser = OptionParser()
    parser.add_option("--pip", dest="pip", default=ROBOT_IP)
    parser.add_option("--pport", dest="pport", type="int", default=ROBOT_PORT)
    (opts, args_) = parser.parse_args()

    if not os.path.exists(LOCAL_TEMP_DIR):
        os.makedirs(LOCAL_TEMP_DIR)

    myBroker = ALBroker("myBroker", "0.0.0.0", 0, opts.pip, opts.pport)
    global bridge
    bridge = PepperBridge("PepperBridge", opts.pip, opts.pport)

    # Create proxies
    tracker = ALProxy("ALTracker", opts.pip, opts.pport)
    face_detection = ALProxy("ALFaceDetection", opts.pip, opts.pport)
    motion = ALProxy("ALMotion", opts.pip, opts.pport)
    posture_proxy = ALProxy("ALRobotPosture", opts.pip, opts.pport)
    idle = ALProxy("ALAutonomousLife", opts.pip, opts.pport)

    print("[main] Starting scenario...")

    # Start face tracking
    idle.setState("solitary")
    start_face_tracking(tracker, face_detection)
    
    # Example usage
    start_text = u"Merhaba!"
    local_start_wav = os.path.join(LOCAL_TEMP_DIR, "start_{}.wav")
    ok = download_tts_to_file(start_text, local_start_wav)
    if ok:
        scp_and_play(local_start_wav, "start_{}.wav", bridge.audio_player)

    launchAndStopBehavior(managerProxy, "animations/Stand/BodyTalk/Speaking/BodyTalk_4")

    start_text = u"Benim adım Deniz. Bugün yaratıcı fikirler üretmeye çalışacağız. Hazır mısın?"
    local_start_wav = os.path.join(LOCAL_TEMP_DIR, "start_{}.wav")
    ok = download_tts_to_file(start_text, local_start_wav)
    if ok:
        scp_and_play(local_start_wav, "start_{}.wav", bridge.audio_player)
    stopBehavior(managerProxy, "animations/Stand/BodyTalk/Speaking/BodyTalk_4")

    time.sleep(2)
    launchAndStopBehavior(managerProxy, "animations/Stand/Gestures/Yes_1")

    start_text = u"Süper! O zaman başlayalım!"
    local_start_wav = os.path.join(LOCAL_TEMP_DIR, "start_{}.wav")
    ok = download_tts_to_file(start_text, local_start_wav)
    if ok:
        scp_and_play(local_start_wav, "start_{}.wav", bridge.audio_player)
    stopBehavior(managerProxy, "animations/Stand/Gestures/Yes_1")

    # Example 2-object scenario
    objects = [u"kalem", u"plastik şişe"]
    for idx, obj_name in enumerate(objects):
        print(u"\n--- Starting object #{}: {} ---".format(idx + 1, obj_name))

        launchAndStopBehavior(managerProxy, "animations/Stand/BodyTalk/Listening/Listening_2")
        current_instruction = u"Şimdi {} nesnesi. 3 dakikan var. Ne yapabiliriz?".format(obj_name)
        local_intro_wav = os.path.join(LOCAL_TEMP_DIR, "intro_{}.wav".format(idx))
        ok = download_tts_to_file(current_instruction, local_intro_wav)
        if ok:
            scp_and_play(local_intro_wav, "intro_{}.wav".format(idx), bridge.audio_player)
        stopBehavior(managerProxy, "animations/Stand/BodyTalk/Listening/Listening_2")
        start_time = time.time()
        last_speech_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed >= 180.0:  # 3 minutes
                # Politely interrupt
                launchAndStopBehavior(managerProxy, "animations/Stand/Gestures/Enthusiastic_2" )
                end_text = u"Zaman doldu. {} için yeterince fikir ürettik!".format(obj_name)
                local_end_wav = os.path.join(LOCAL_TEMP_DIR, "end_{}.wav".format(idx))
                if download_tts_to_file(end_text, local_end_wav):
                    scp_and_play(local_end_wav, "end_{}.wav".format(idx), bridge.audio_player)
                stopBehavior(managerProxy, "animations/Stand/Gestures/Enthusiastic_2")
                break

            # --- The only line changed:  pick random idle messages ---
            if (time.time() - last_speech_time) >= 15.0:
                # Randomly pick from the 3 idle messages
                idle_text = random.choice(IDLE_MESSAGES)
                local_idle_wav = os.path.join(LOCAL_TEMP_DIR, "idle_{}.wav".format(idx))
                if download_tts_to_file(idle_text, local_idle_wav):
                    scp_and_play(local_idle_wav, "idle_{}.wav".format(idx), bridge.audio_player)
                    print("[Idle] Played idle message: '%s'" % idle_text)
                    last_speech_time = time.time()
                else:
                    print("[Idle] Failed to generate or play idle audio.")

            # Record short audio (3s), then sum to mono
            local_record_file = os.path.join(LOCAL_TEMP_DIR, "user_{}.wav".format(idx))
            bridge.record_audio(local_record_file, duration=3)

            # Send to STT
            response_tuple = post_audio_for_stt(local_record_file, current_instruction=current_instruction)
            if response_tuple is None:
                print("[main] No response from STT server. Skipping iteration.")
                continue

            r, net_delay = response_tuple
            if r is None:
                print("[main] STT request failed. Skipping.")
                continue

            # If STT/LLM/TTS took >3s, insert a filler phrase & gesture
            if net_delay > 5.0:
                behaviors = [
                    "animations/Stand/Waiting/ScratchHead_1",
                    "animations/Stand/Gestures/Thinking_5",
                    "animations/Stand/Gestures/Thinking_6"
                ]
                selected_behavior = random.choice(behaviors)
                try:
                    managerProxy.runBehavior(selected_behavior)
                    print("[Filler] Running behavior: {}".format(selected_behavior))
                except Exception as e:
                    print("[Filler] Error running behavior: {}".format(e))

                filler_idx = int((time.time() * 1000)) % len(FILLER_PHRASES)
                filler_text = FILLER_PHRASES[filler_idx]
                local_filler_wav = os.path.join(LOCAL_TEMP_DIR, "filler_{}.wav".format(idx))
                if download_tts_to_file(filler_text, local_filler_wav):
                    scp_and_play(local_filler_wav, "filler_{}.wav".format(idx), bridge.audio_player)

                try:
                    managerProxy.stopBehavior(selected_behavior)
                    print("[Filler] Stopped behavior: {}".format(selected_behavior))
                except Exception as e:
                    print("[Filler] Error stopping behavior: {}".format(e))

            # Check STT result
            if r.status_code == 200:
                js = r.json()
                recognized_text = js.get("recognized_text", "")
                chat_response = js.get("chatgpt_response", "")
                wav_b64 = js.get("wav_base64", None)

                if recognized_text.strip():
                    last_speech_time = time.time()
                    print("[main] User said: {}".format(recognized_text.encode('utf-8')))

                # If server returns TTS audio
                if wav_b64:
                    wav_data = base64.b64decode(wav_b64)
                    local_response_wav = os.path.join(LOCAL_TEMP_DIR, "response_{}.wav".format(idx))
                    with open(local_response_wav, "wb") as f:
                        f.write(wav_data)

                    audio_duration = get_wav_duration(local_response_wav)
                    print("[main] Audio duration: {:.2f} seconds".format(audio_duration))
                    launch_random_gestures(managerProxy, SPEAKING_GESTURES, audio_duration)
                    stopBehavior(managerProxy, "animations/Stand/BodyTalk/Speaking/BodyTalk_8")
                    stopBehavior(managerProxy, "animations/Stand/BodyTalk/Speaking/BodyTalk_10")
                    stopBehavior(managerProxy, "animations/Stand/BodyTalk/Speaking/BodyTalk_1")
                    stopBehavior(managerProxy, "animations/Stand/BodyTalk/Speaking/BodyTalk_14")
                    stopBehavior(managerProxy, "animations/Stand/BodyTalk/Speaking/BodyTalk_20")
                    # Play the response
                    scp_and_play(local_response_wav, "response_{}.wav".format(idx), bridge.audio_player)

                    last_speech_time = time.time()
                    
            else:
                print("[main] /listenUser error code:", r.status_code, r.text)
                continue

    # End scenario
    wave_hand(posture_proxy, motion, hand="right", speed=2)
    final_text = u"Teşekkür ederim! Görevi tamamladık. Çok yaratıcı fikirler bulduk!"
    local_final = os.path.join(LOCAL_TEMP_DIR, "final.wav")
    if download_tts_to_file(final_text, local_final):
        scp_and_play(local_final, "final.wav", bridge.audio_player)

    print("[main] Exiting scenario.")
    myBroker.shutdown()
    sys.exit(0)

if __name__ == "__main__":
    main()
