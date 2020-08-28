import sounddevice, time
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io import AudioFileClip
import imageio
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import numpy as np
from Real_Time_Voice_Cloning.synthesizer import hparams as syn_params
sample_rate = syn_params.hparams.sample_rate


def play(wav, pause_processes=True):
    if isinstance(wav, str):
        wav = load(wav)
    sounddevice.stop()
    sounddevice.play(wav, sample_rate)
    if pause_processes:
        time.sleep(wav.shape[0] / sample_rate + 1)


def record(duration, pause_processes=True):
    # sounddevice.stop()
    try:
        wav = sounddevice.rec(duration * sample_rate, sample_rate, 1)
        if pause_processes:
            time.sleep(duration)
        wav = wav.reshape(wav.shape[0])
    except Exception as e:
        print(e)
        print("Could not record anything. Is your recording device enabled?")
        print("Your device must be connected before you start the toolbox.")
        return None
    return wav


def save(wav, filepath):
    # sf.write(filepath, wav, sample_rate)
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    # proposed by @dsmiller
    wavfile.write(filepath, sample_rate, wav.astype(np.int16))


def load(filepath, get_sample=False):
    wav, sr = librosa.core.load(filepath, sr=sample_rate)
    if get_sample:
        return wav, sr
    return wav


def mp3_to_wav(mp3_path):
    sound = AudioSegment.from_mp3(mp3_path)
    new_path = mp3_path[:-4] + ".wav"
    sound.export(new_path, format="wav")
    return load(new_path)


def audio_from_mp4(mp4_file):
    vid = VideoFileClip(mp4_file)
    return vid.audio


def create_mp4(video, audio, save_path=None):
    audio = AudioFileClip(audio) if audio is not None else None
    fps = video.shape[0] / audio.duration
    imageio.mimsave("backend_files/mute.mp4", video)
    video = VideoFileClip("backend_files/mute.mp4", audio=False)
    video.set_audio(audio)
    if save_path is not None:
        video.write_videofile(save_path, fps=fps)  # , codec='mpeg4')
    return video
