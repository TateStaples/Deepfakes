import numpy as np


# audio section
def text_to_speech(text, driving_audio, save_audio_path=None, save_voice_path=None, play=False):  # around 10 seconds
    """
    Takes a path to a text document and creates a voice saying those words in target's voice
    Algorithm: modified Real-Time Voice Cloning Toolbox
    source: https://github.com/CorentinJ/Real-Time-Voice-Cloning
    :param text: path to the words to be spoken or a string
    :param driving_audio: single or list of .wav audio samples
    :param save_audio_path: optional path to save to the voice file to
    :param save_voice_path: optional path to save your voice embedding (speeds future transfers)
    :param play: should it be played at the end
    :return: wav file
    """
    import text_to_audio
    import utils
    # list of samples or pre-made voice embedding
    if isinstance(driving_audio, list) or isinstance(driving_audio, str) and driving_audio[-4:] == ".npy":
        wav, voice = text_to_audio.generate_audio(text, driving_audio)
    # single voice sample
    else:
        wav, voice = text_to_audio.generate_audio(text, [driving_audio])
    if play:
        utils.play(wav)
    if save_audio_path is not None:
        utils.save(wav, save_audio_path)
    if save_voice_path is not None and voice is not None:
        np.save(save_voice_path, voice)
    return wav


def audio_stylize(base_audio, driving_audio, result_path):  # 5 seconds
    """
    Take a random voice and convert into target
    Algorithm: One-shot Voice Conversion by Separating Speaker and Content Representations with Instance Normalization
    source: https://github.com/jjery2243542/adaptive_voice_conversion.git
    :param base_audio: someone speaking the words
    :param driving_audio: example of the target voice saying things (recommend large sample size)
    :param result_path: optional path to save result
    :return: audio in the target voice (.wav)
    """
    import audio_transform
    return audio_transform.transorm_audio(base_audio, driving_audio, result_path)


# video
def audio_to_image(driving_audio, base_img, result_path=None):  # around 40 seconds
    """
    Use previously trained videos to imitate how person would say words
    Algorithm: Speech-Driven Facial Animation
    source: https://github.com/DinoMan/speech-driven-animation
    :param driving_audio: The audio you want the person to say
    :param base_img: A picture of who you want to speak the words
    :param result_path: optional path for where to store the result
    :return: video, audio
    """
    import audio_to_vid
    return audio_to_vid.generate_video(base_img, driving_audio, result_path)


def deepfake_video(base_video, driving_img, result_path=None):  # around 15 minutes
    """
    Take a video and an image and generate a new video with the faces swapped
    Algorithm: First Order Motion Model for Image Animation
    source: https://github.com/AliaksandrSiarohin/first-order-model
    :param base_video: video of random movements
    :param driving_img: target that will perform the motions
    :param result_path: optional path to save the resulting mp4
    :return: video file of target doing the actions and the fps of the video
    """
    import pic_to_vid
    video, fps = pic_to_vid.demo_video(driving_img, base_video, result_path, auto_crop=False)
    return video, fps


# compound
def text_to_vid(text, driving_audio, driving_img, result_path=None):
    """
    Creates a video of a person saying input text.
    :param text: a string or txt file path for what you want the target to say
    :param driving_audio: a sample of the targets voice (.wav)
    :param driving_img: an image of the target
    :param result_path: optional save path for the result
    :return: video, audio
    """
    synthesized_audio = text_to_speech(text, driving_audio)
    return audio_to_image(synthesized_audio, driving_img, result_path)


def imitate(base_video, driving_audio, driving_img, result_path):
    """
    Takes a video of one person saying something and replaces it with someone else
    :param base_video: path to the original video (.mp4)
    :param driving_audio: path to the voice of who you want to speak (.wav or .mp3)
    :param driving_img: path to the img you want to copy
    :param result_path: path to save the resulting video
    :return: VideoFileClip (moviepy module) of the video
    """
    import utils
    base_audio = utils.audio_from_mp4(base_video)
    new_audio = audio_stylize(base_video, driving_audio)
    new_video = deepfake_video(base_video, driving_img)
    return utils.save_mp4(new_video, new_audio, result_path)


if __name__ == '__main__':
    pass
