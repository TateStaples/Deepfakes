from adaptive_voice_conversion.inference import Inferencer
from utils import sample_rate
import yaml


class Blank: pass

def transorm_audio(content, style, output):
    with open("adaptive_voice_conversion/config.yaml") as f:
        config = yaml.load(f)
    args = Blank()
    args.attr = "adaptive_voice_conversion/attr.pkl"
    args.config = config
    args.model = "adaptive_voice_conversion/vctk_model.ckpt"
    args.source = content
    args.target = style
    args.output = output
    args.sample_rate = sample_rate

    inferencer = Inferencer(config=config, args=args)
    return inferencer.inference_from_path(output is not None)