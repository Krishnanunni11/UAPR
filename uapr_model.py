import os
import csv
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import whisper

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
LABELS_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
import urllib.request
labels_txt = urllib.request.urlopen(LABELS_URL)
import io
yamnet_labels = [row[2] for row in csv.reader(io.StringIO(labels_txt.read().decode('utf-8')))][1:]
whisper_model = whisper.load_model("base")

HIGH_PRIORITY_KEYWORDS = [
    "evacuate", "fire", "earthquake", "emergency", "immediately", "explosion", "danger", "help",
    "ambulance", "evacuation", "bomb", "critical"
]
MEDIUM_PRIORITY_KEYWORDS = [
    "warning", "caution", "attention", "incident", "alert", "notice", "hazard", "drill", "police"
]

def get_sound_events(audio_file, top_k=10, print_all=True):
    wav, sr = librosa.load(audio_file, sr=16000, mono=True)
    waveform = wav.astype(np.float32)
    scores, embeddings, spectrogram = yamnet_model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0).numpy()
    top_indices = np.argpartition(mean_scores, -top_k)[-top_k:]
    detected = [(yamnet_labels[i], float(mean_scores[i])) for i in top_indices]
    detected.sort(key=lambda x: x[1], reverse=True)
    if print_all:
        print("Top Detected Audio Events:")
        for label, score in detected:
            print(f"  {label:35}  {score:.4f}")
    return detected

def text_priority(text):
    t = text.strip().lower()
    if not t:
        return "None"
    if any(kw in t for kw in HIGH_PRIORITY_KEYWORDS):
        return "High"
    if any(kw in t for kw in MEDIUM_PRIORITY_KEYWORDS):
        return "Medium"
    return "Low"

def get_transcribed_priority(audio_file):
    print("Transcribing speech and analyzing priority...")
    result = whisper_model.transcribe(audio_file)
    text = result['text'].strip()
    priority = text_priority(text)
    print(f"  Transcript: {text!r}")
    print(f"  Speech Priority: {priority}")
    return priority, text

def priority_fusion(sound_events, speech_priority, event_threshold_high=0.20, event_threshold_medium=0.10, print_reason=True):
    score = 0
    reason = []
    for cname, conf in sound_events:
        lc = cname.lower()
        if "siren" in lc or "alarm" in lc or "police car" in lc or "ambulance" in lc:
            if conf > event_threshold_high:
                score += 3
                reason.append(f"SIREN/ALARM ({cname} {conf:.2f})")
            elif conf > event_threshold_medium:
                score += 1.2
                reason.append(f"(weak) SIREN/ALARM ({cname} {conf:.2f})")
        elif "announcement" in lc or "loudspeaker" in lc:
            if conf > event_threshold_medium:
                score += 1
                reason.append(f"ANNOUNCEMENT ({cname} {conf:.2f})")
        elif "megaphone" in lc or "yell" in lc:
            if conf > event_threshold_medium:
                score += 0.5
                reason.append(f"SPEECH ALERT ({cname} {conf:.2f})")
    if speech_priority == "High":
        score += 3
        reason.append("Speech priority HIGH")
    elif speech_priority == "Medium":
        score += 1.3
        reason.append("Speech priority MEDIUM")
    final = "Low"
    if score >= 3.5:
        final = "High"
    elif score >= 1.5:
        final = "Medium"
    if print_reason:
        print(f"\nDecision Reasoning: score={score:.2f}; triggers: {', '.join(reason) if reason else 'None'}")
        print(f"--> Final Priority: {final}")
    return final, reason, score

def export_results(audio_file, final_priority, transcript, sound_events, reasoning, score, out_csv='audio_priority_results.csv'):
    headers = [
        'Audio File', 'Final Priority', 'Score', 'Transcript', 'Detected Events', 'Reasoning'
    ]
    event_str = " | ".join([f"{l}:{s:.2f}" for l, s in sound_events])
    reason_str = " | ".join(reasoning)
    row = [
        audio_file,
        final_priority,
        f"{score:.2f}",
        transcript,
        event_str,
        reason_str
    ]
    file_exists = os.path.isfile(out_csv)
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
    print(f"\nResults exported to {out_csv} in readable table format.")

def detect_audio_priority(audio_file, out_csv='audio_priority_results.csv'):
    print(f"\nAnalyzing audio file: {audio_file}")
    sound_events = get_sound_events(audio_file, top_k=10, print_all=True)
    speech_prio, transcript = get_transcribed_priority(audio_file)
    final_priority, reasoning, score = priority_fusion(sound_events, speech_prio, print_reason=True)
    export_results(audio_file, final_priority, transcript, sound_events, reasoning, score, out_csv)
    return final_priority, transcript

if __name__ == "__main__":
    print("==== UNIVERSAL AUDIO PRIORITY RANKER ====")
    print("Supported: mono/stereo .wav/.mp3 (.flac if ffmpeg present)")
    audio_path = input("Enter path to audio file: ").strip()
    detect_audio_priority(audio_path)
