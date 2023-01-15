import time
from pathlib import Path
from hashlib import sha1

import soundfile as sf
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model

output_folder = Path('synthesized_wavs')
if not output_folder.exists():
    output_folder.mkdir(parents=True, exist_ok=True)


class ModelDontExist(Exception):
    pass


def available_models() -> list[str]:
    models_path = Path('models')
    if not models_path.exists():
        raise ModelDontExist('No available voice models')
    return [folder.name for folder in models_path.iterdir() if folder.exists()]


def process_text(text: str, model: str) -> tuple[Path, float]:
    # specify path and setup vocoder
    vocoder_checkpoint = f'models/{model}/vocoder/checkpoint-400000steps.pkl'
    vocoder = load_model(vocoder_checkpoint).to("cpu").eval()
    vocoder.remove_weight_norm()

    # specify path to the model
    model_path = f'models/{model}/tts_train_raw_char/train.loss.ave_5best.pth'

    # setup tts
    text2speech = Text2Speech(
        model_file=model_path,
        device="cpu",
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=True,
        backward_window=1,
        forward_window=3,
    )
    text2speech.spc2wav = None  # Disable griffin-lim

    start = time.time()
    with torch.no_grad():
        mel = text2speech(text.lower())
        wav = vocoder.inference(mel['feat_gen'])

    name = f'tts-{model}-{sha1(text.encode()).hexdigest()}.wav'
    output_wav = output_folder / name
    rtf = round((time.time() - start) / (len(wav) / text2speech.fs), 3)  # processing time to sample length ratio

    sf.write(output_wav, wav.numpy(), text2speech.fs, "PCM_16")

    return output_wav, rtf


def check_voice_cache(text: str, model: str) -> Path | None:
    name = f'tts-{model}-{sha1(text.encode()).hexdigest()}.wav'
    file_path = Path(output_folder) / name
    if file_path.exists():
        return file_path


if __name__ == '__main__':
    models = available_models()
    sample_text = 'Менің атым Қожа болады. Ал сіздің атыңыз қалай?'
    process_text(sample_text, next(iter(models)))
