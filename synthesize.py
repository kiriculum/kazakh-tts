import time
from pathlib import Path

import soundfile as sf
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model

fs = 22050

voices = []
# voices = ['female1']
sample_text = 'Менің атым Қожа болады. Ал сіздің атыңыз қалай?'

for voice in voices:
    # specify the path to vocoder's checkpoint
    vocoder_checkpoint = f'exp/{voice}/vocoder/checkpoint-400000steps.pkl'
    vocoder_config = f'exp/{voice}/vocoder/config.yml'
    vocoder = load_model(vocoder_checkpoint).to("cpu").eval()
    vocoder.remove_weight_norm()

    # specify path to the main model(transformer/tacotron2/fastspeech) and its config file
    config_file = f'exp/{voice}/tts_train_raw_char/config.yaml'
    model_path = f'exp/{voice}/tts_train_raw_char/train.loss.ave_5best.pth'

    # setup tts
    text2speech = Text2Speech(
        # config_file,
        model_file=model_path,
        # vocoder_file=vocoder_checkpoint,
        # vocoder_config=vocoder_config,
        device="cpu",
        # Only for Tacotron 2
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        use_att_constraint=True,
        backward_window=1,
        forward_window=3,
        # Only for FastSpeech & FastSpeech2
        speed_control_alpha=1.0,
    )
    text2speech.spc2wav = None  # Disable griffin-lim

    start = time.time()
    with torch.no_grad():
        res = text2speech(sample_text.lower())
        wav = vocoder.inference(res['feat_gen'])
        rtf = (time.time() - start) / (len(wav) / text2speech.fs)
        print(f'RTF = {rtf:04f}')

    # here all of your synthesized audios will be saved
    folder_to_save, wav_name = 'synthesized_wavs', f'example-{voice}.wav'

    Path(folder_to_save).mkdir(parents=True, exist_ok=True)
    sf.write(folder_to_save + f"/{wav_name}", wav.numpy(), text2speech.fs, "PCM_16")
