import os
import time
import threading
import numpy as np
import pyaudio
import scipy.io.wavfile as wav
import tkinter as tk
from tkinter import scrolledtext, ttk
import assemblyai as aai
import openai

FS = 16000
CHANNELS = 1
FORMAT = pyaudio.paFloat32
EPOCH_DURATION = 5.0
OVERLAP_PERCENTAGE = 20
STEP_DURATION = EPOCH_DURATION * (1 - OVERLAP_PERCENTAGE / 100)
SAMPLES_PER_EPOCH = int(EPOCH_DURATION * FS)
SAMPLES_PER_STEP = int(STEP_DURATION * FS)
TEMP_FILENAME = "epoch.wav"

EPOCHS_PER_QUERY = 2
LOOKBACK_SECONDS = 120

with open("assemblyaikey.txt", "r") as f:
    aai_key = f.read().strip()
with open("openaikey.txt", "r") as f:
    openai_key = f.read().strip()
    
aai.settings.api_key = aai_key
TRANSCRIPTION_CONFIG = aai.TranscriptionConfig(speaker_labels=True)
openai.api_key = openai_key

def save_epoch_to_wav(audio_data: np.ndarray, filename: str, samplerate: int):
    """
    Save a NumPy array as a WAV file in int16 format.

    Parameters
    ----------
    audio_data : np.ndarray
        Audio data array.
    filename : str
        Name of the file to save.
    samplerate : int
        Sampling rate.
    """
    if np.max(np.abs(audio_data)) > 0:
        scaled = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
    else:
        scaled = np.int16(audio_data)
    wav.write(filename, samplerate, scaled)

def analyze_sentiment(text: str) -> str:
    """
    Analyze sentiment from text using OpenAI and return a one-word summary.

    Parameters
    ----------
    text : str
        Text to analyze.

    Returns
    -------
    str
        One-word sentiment summary.
    """
    prompt = f"Analyze the sentiment in this text and summarize it in one word: {text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        sentiment = response.choices[0].message.content.strip()
    except Exception as e:
        sentiment = f"Error: {e}"
    return sentiment

class AudioRecorder:
    """
    Manage PyAudio stream for capturing audio.
    """
    def __init__(self, rate=FS, channels=CHANNELS, fmt=FORMAT, frames_per_buffer=1024):
        self.rate = rate
        self.channels = channels
        self.fmt = fmt
        self.frames_per_buffer = frames_per_buffer
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None

    def start(self):
        """
        Open audio input stream.
        """
        self.stream = self.audio_interface.open(
            format=self.fmt,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer
        )

    def read(self, num_frames: int) -> np.ndarray:
        """
        Read num_frames samples from the stream.

        Parameters
        ----------
        num_frames : int
            Number of samples to read.

        Returns
        -------
        np.ndarray
            Array of audio samples.
        """
        frames = []
        frames_read = 0
        while frames_read < num_frames:
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow=False)
            chunk = np.frombuffer(data, dtype=np.float32)
            frames.append(chunk)
            frames_read += len(chunk)
        audio_data = np.concatenate(frames)
        return audio_data[:num_frames]

    def stop(self):
        """
        Stop and close the audio stream.
        """
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.audio_interface.terminate()

class RecorderThread(threading.Thread):
    """
    Continuous recording, transcription, and data handling thread.
    """
    def __init__(self, transcription_callback, gpt_model="gpt-3.5-turbo", *args, **kwargs):
        """
        Parameters
        ----------
        transcription_callback : callable
            Function to call with transcription results.
        gpt_model : str
            OpenAI GPT model to use for queries.
        """
        super().__init__(*args, **kwargs)
        self.transcription_callback = transcription_callback
        self.gpt_model = gpt_model
        self._running = threading.Event()
        self._running.set()
        self.audio_buffer = np.empty((0,), dtype=np.float32)
        self.recorder = AudioRecorder()
        self.conversation_data = []
        self.epoch_count = 0
        self.gpt_responses = []
        self.start_time = None

    def transcribe_and_save(self, file_path: str):
        """
        Transcribe the audio file, store utterances in conversation_data, and return a readable string.

        Parameters
        ----------
        file_path : str
            Path to the WAV file.

        Returns
        -------
        str
            Human-readable transcription summary.
        """
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_path, config=TRANSCRIPTION_CONFIG)
        if transcript.status == aai.TranscriptStatus.error:
            return f"Transcription error: {transcript.error}"
        current_time = time.time() - self.start_time
        result_text = "Transcription complete:\n"
        for utterance in transcript.utterances:
            speaker = f"Speaker {utterance.speaker}"
            text = utterance.text
            real_timestamp = current_time + (utterance.start / 1000.0)
            sentiment = analyze_sentiment(text)
            self.conversation_data.append({
                "speaker": speaker,
                "text": text,
                "timestamp": real_timestamp,
                "sentiment": sentiment
            })
            result_text += f"{speaker}: {text}\nSentiment: {sentiment}\n"
        return result_text

    def gather_recent_conversation_string(self):
        """
        Gather conversation data from the last LOOKBACK_SECONDS and return a merged string.

        Returns
        -------
        str
            String with speaker, text, timestamp, sentiment for recent utterances.
        """
        cutoff_time = time.time() - self.start_time - LOOKBACK_SECONDS
        recent_rows = [row for row in self.conversation_data if row["timestamp"] >= cutoff_time]
        recent_conv_str = ""
        for row in recent_rows:
            recent_conv_str += (
                f"Speaker: {row['speaker']} | "
                f"Timestamp: {round(row['timestamp'], 2)} | "
                f"Sentiment: {row['sentiment']} | "
                f"Text: {row['text']}\n"
            )
        return recent_conv_str

    def query_gpt_model(self, instructions: str, conversation_str: str):
        """
        Query the GPT model with merged instructions and conversation string.

        Parameters
        ----------
        instructions : str
            Prompt instructions read from file.
        conversation_str : str
            Recent conversation string.

        Returns
        -------
        str
            GPT response.
        """
        merged_prompt = instructions + "\n" + conversation_str
        try:
            response = openai.ChatCompletion.create(
                model=self.gpt_model,
                messages=[{"role": "user", "content": merged_prompt}],
                temperature=0.0,
            )
            gpt_reply = response.choices[0].message.content.strip()
        except Exception as e:
            gpt_reply = f"Error: {e}"
        return gpt_reply

    def run(self):
        """
        Record audio in steps, process epochs, and query GPT periodically.
        """
        self.start_time = time.time()
        self.recorder.start()
        try:
            while self._running.is_set():
                new_audio = self.recorder.read(SAMPLES_PER_STEP)
                self.audio_buffer = np.concatenate((self.audio_buffer, new_audio))
                if len(self.audio_buffer) >= SAMPLES_PER_EPOCH:
                    epoch_data = self.audio_buffer[-SAMPLES_PER_EPOCH:]
                    save_epoch_to_wav(epoch_data, TEMP_FILENAME, FS)
                    transcription_result = self.transcribe_and_save(TEMP_FILENAME)
                    self.transcription_callback(transcription_result)
                    if os.path.exists(TEMP_FILENAME):
                        os.remove(TEMP_FILENAME)
                    self.audio_buffer = self.audio_buffer[-SAMPLES_PER_EPOCH:]
                    self.epoch_count += 1
                    if self.epoch_count % EPOCHS_PER_QUERY == 0:
                        with open("prompt_instructions.txt", "r") as f:
                            prompt_instructions = f.read().strip()
                        recent_conv_str = self.gather_recent_conversation_string()
                        if recent_conv_str.strip():
                            gpt_result = self.query_gpt_model(prompt_instructions, recent_conv_str)
                            self.gpt_responses.append(gpt_result)
                            self.transcription_callback(f"GPT Response:\n{gpt_result}")
                time.sleep(0.1)
        except Exception as e:
            self.transcription_callback(f"Error in recording thread: {e}")
        finally:
            self.recorder.stop()

    def stop(self):
        """
        Signal the thread to stop.
        """
        self._running.clear()

class TranscriptionApp:
    """
    GUI for live transcription, sentiment analysis, and GPT queries.
    """
    def __init__(self, master):
        """
        Parameters
        ----------
        master : tk.Tk
            The root window.
        """
        self.master = master
        master.title("Live Transcription & Sentiment Analysis")
        master.geometry("600x500")
        self.start_button = ttk.Button(master, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=10)
        self.stop_button = ttk.Button(master, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)
        self.text_area = scrolledtext.ScrolledText(master, wrap=tk.WORD, height=20)
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.recorder_thread = None

    def append_text(self, message: str):
        """
        Append text to the GUI text area.

        Parameters
        ----------
        message : str
            Message to append.
        """
        def update_text():
            self.text_area.insert(tk.END, message + "\n")
            self.text_area.see(tk.END)
        self.master.after(0, update_text)

    def transcription_callback(self, transcription_text: str):
        """
        Callback for transcription results.

        Parameters
        ----------
        transcription_text : str
            The transcription and sentiment analysis or GPT output.
        """
        self.append_text(transcription_text)

    def start_recording(self):
        """
        Start the recording thread.
        """
        self.append_text("Starting recording...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.recorder_thread = RecorderThread(transcription_callback=self.transcription_callback)
        self.recorder_thread.daemon = True
        self.recorder_thread.start()

    def stop_recording(self):
        """
        Stop the recording thread.
        """
        self.append_text("Stopping recording...")
        self.stop_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.NORMAL)
        if self.recorder_thread:
            self.recorder_thread.stop()
            self.recorder_thread.join()
            self.recorder_thread = None
        self.append_text("Recording stopped.")

def main():
    """
    Entry point of the application.
    """
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
