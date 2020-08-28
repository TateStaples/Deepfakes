import speech_driven_animation.sda as sda
import utils


def generate_video(target_image, audio_path, save_path):
    animator = sda.VideoAnimator()  # Instantiate the animator
    fs = None if isinstance(audio_path, str) else utils.sample_rate
    video, audio_file = animator(target_image, audio_path, fs=fs, aligned=False)
    # print(video.shape)
    if save_path is not None:
        animator.save_video(video, audio_file, save_path)
    return video, audio_file