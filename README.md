## UAPR
# Universal Audio Priority Ranker

**AI-Powered Audio Priority Detection System**  
Developed during **HackQuest’25** 🏆  

---

## 🧠 Overview
The **Universal Audio Priority Ranker (UAPR)** is an intelligent AI model that listens to an audio file and automatically determines its **priority level** — **High**, **Medium**, or **Low** — based on the **sound events** and **spoken content** within the audio.  

It’s designed for smart safety systems like **emergency detection**, **surveillance**, or **autonomous vehicles**, where identifying urgent sounds or commands can save lives.

---

## ⚙️ Features

✅ **Sound Event Detection** – Uses Google’s pre-trained [YAMNet](https://tfhub.dev/google/yamnet/1) model to recognize key sounds like sirens, alarms, police cars, and megaphones.  
✅ **Speech Transcription** – Uses [OpenAI Whisper](https://github.com/openai/whisper) to convert spoken audio into text.  
✅ **Keyword-Based Priority** – Detects critical terms such as “fire”, “evacuate”, “emergency”, and assigns corresponding priority.  
✅ **Fusion Algorithm** – Combines both sound and speech analysis to generate a final urgency score.  
✅ **CSV Export** – Stores results (priority level, reasoning, detected events, transcript, and score) in a structured CSV format.

---

## 🏗️ System Architecture

🎧 Audio Input
↓
🔍 Sound Event Detection (YAMNet)
↓
🗣️ Speech-to-Text (Whisper)
↓
⚙️ Priority Fusion Logic
↓
📊 Priority Classification + CSV Export



---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|----------------|
| Programming Language | Python |
| Sound Event Model | TensorFlow Hub – YAMNet |
| Speech-to-Text | OpenAI Whisper |
| Audio Processing | Librosa |
| Data Handling | NumPy, CSV |
| Frameworks | TensorFlow, TensorFlow Hub |

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Krishnanunni11/UAPR.git
cd AudioPriorityRanker
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
tensorflow
tensorflow_hub
numpy
librosa
openai-whisper
```

### 3️⃣ Run the model

```bash
python audio_priority_ranker.py
```

When prompted, enter the path to your `.wav` or `.mp3` file.

---

## 🧮 Example Output

```
Analyzing audio file: siren_alert.wav
Top Detected Audio Events:
  Siren                             0.87
  Alarm                             0.74
Transcribing speech and analyzing priority...
  Transcript: 'Evacuate immediately!'
  Speech Priority: High

Decision Reasoning: score=6.20; triggers: SIREN/ALARM (Siren 0.87), Speech priority HIGH
--> Final Priority: High
```

📁 **Results Exported To:** `audio_priority_results.csv`

| Audio File      | Final Priority | Score | Transcript            | Detected Events | Reasoning                    |
| --------------- | -------------- | ----- | --------------------- | --------------- | ---------------------------- |
| siren_alert.wav | High           | 6.20  | Evacuate immediately! | Siren:0.87      | SIREN + Speech Priority HIGH |

---

## 💡 Use Cases

🔸 Emergency detection systems
🔸 Smart city monitoring
🔸 IoT-based safety networks
🔸 Autonomous vehicle sound analysis
🔸 Real-time alert prioritization

---

## 🧑‍💻 Developer

**👤 Krishnanunni H Pillai**
B.Tech Student | AI & IoT Enthusiast
🔗 [LinkedIn](www.linkedin.com/in/krishnanunni-h-pillai-b87448327)

---

## 📜 License

This project is licensed under the **MIT License** — you’re free to use, modify, and distribute with proper attribution.

---

## ⭐ Support

If you found this project interesting, please give it a ⭐ on GitHub and share it!
Your feedback and contributions are always welcome. 😊


