# things for models
from Real_Time_Voice_Cloning.encoder import inference as encoder
from Real_Time_Voice_Cloning.synthesizer.inference import Synthesizer
from Real_Time_Voice_Cloning.vocoder import inference as vocoder

# other package file
from Real_Time_Voice_Cloning.synthesizer import hparams as syn_params
from Real_Time_Voice_Cloning.toolbox.utterance import Utterance

# normal
import numpy as np
import utils

sample_rate = syn_params.hparams.sample_rate
synthesizer = None
current_synthesized_model = None


def get_synthesizer(path=""):  # create spectrogram for voice
    if synthesizer is None:
        checkpoints_dir = path + "/taco_pretrained"
        return Synthesizer(checkpoints_dir, low_mem=True, verbose=False)
    return synthesizer


def generate_spectrogram(text, utterance):
    texts = text.split("\n")
    embed = utterance.embed
    embeds = np.stack([embed] * len(texts))
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)

    # self.ui.draw_spec(spec, "generated")
    # self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
    return spec, breaks


def decode_spectrogram(spec, breaks=False):
    wav = vocoder.infer_waveform(spec)

    # Add breaks
    if breaks:
        b_ends = np.cumsum(np.array(breaks) * syn_params.hparams.hop_size)
        b_starts = np.concatenate(([0], b_ends[:-1]))
        wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
        breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
        wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    # trim silences
    wav = encoder.preprocess_wav(wav)

    # Play it
    wav = wav / np.abs(wav).max() * 0.97
    return wav


def create_utterance(wavs):
    amount_of_samples = len(wavs)
    embeds = []
    # Compute the mel spectrogram
    spec = Synthesizer.make_spectrogram(wavs[0])
    # self.ui.draw_spec(spec, "current")

    for wav in wavs:
        # Compute the embedding
        encoder_wav = encoder.preprocess_wav(wav)
        # embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        embed = encoder.embed_utterance(encoder_wav, return_partials=False)
        embeds.append(embed)
    avg_embed = sum(embeds) / amount_of_samples
    speaker_name = "audio_sample"
    name = speaker_name + "_rec_%05d"
    # Add the utterance
    return Utterance(name, speaker_name, wavs[0], spec, avg_embed, None, False)


def generate_audio(text, audio_samples):
    '''
    Return an audio file of a text in the voice of some utterances from the same person
    :param text: text file or string with line breaks to indicate pauses
    :param audio_samples: paths to any audio sample of .wav format (5-12 seconds)
    :return: audio file of .wav format
    '''
    # todo: import the pretrained models
    global current_synthesized_model, synthesizer
    encoder.load_model("Real_Time_Voice_Cloning/pretrained/encoder/saved_models/pretrained.pt", "cpu")  # what is this used for
    vocoder.load_model("Real_Time_Voice_Cloning/pretrained/vocoder/saved_models/pretrained.pt", verbose=False)
    # todo: figure out how the multiple utterances work
    synthesizer = get_synthesizer("Real_Time_Voice_Cloning/pretrained/synthesizer/saved_models/logs-pretrained")
    if len(text) > 4 and text[-4:] == ".txt":  # check if file
        words = ""
        with open(text) as file:
            for line in file:
                words += line
        text = words
        del words
    if isinstance(audio_samples, str):
        utterance = Utterance("name", "speaker_name", None, None, np.load(audio_samples), None, None)
    else:
        utterance = create_utterance(audio_samples)
    current_synthesized_model = generate_spectrogram(text, utterance)
    audio_file = decode_spectrogram(*current_synthesized_model)
    return audio_file, utterance.embed


if __name__ == '__main__':
    sample_rate = syn_params.hparams.sample_rate
    while True:
        input("Hit enter to record:")
        wav = utils.record(sample_rate, 5)
        input("Hit enter to play")
        utils.play(wav, sample_rate)
        print(wav.shape)

