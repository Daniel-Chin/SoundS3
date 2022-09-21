from os import path
import sys
sys.path.append(path.abspath(path.join(
    path.dirname(__file__), '../..', 
)))

from SoundS3.shared import DEVICE, HAS_CUDA
from normal_rnn import Conv2dGruConv2d, LAST_H, LAST_W, IMG_CHANNEL, CHANNELS
from train_config import CONFIG
from tkinter import *
import os
import torch
from trainer_symmetry import save_spectrogram, tensor2spec, norm_log2, norm_log2_reverse, LOG_K
from SoundS3.SoundDataLoader import SoundDataLoader

WAV_PATH = '../../../pitch_linearity_eval/datasets/raising_12_wav_SF-GS/'
MODELS = [
    ('vae_aug_0', '../../../trained_models/sound/VAE_noSymm_cp200000(ablation).pt'), 
    ('vae_aug_4', '../../../trained_models/sound/VAE_symm4_cp150000.pt'), 
    ('ae_aug_4', '../../../trained_models/sound/AE_symm4_cp160000.pt'), 
]
OUTPUT = './zLattice/'

def loadModels():
    models = []
    for name, model_path in MODELS:
        model = Conv2dGruConv2d(CONFIG)
        model.eval()
        assert os.path.exists(model_path)
        model.load_state_dict(torch.load(
            model_path, map_location=DEVICE, 
        ))
        models.append((name, model))
        print('loaded model', name)
    return models

def evalModel(model_name, model, data, data_file_name):
    z, mu, logvar = model.batch_seq_encode_to_z(data)
    pitch_seq = mu[0, 0:7, 0].detach()
    from console import console
    console({**globals(), **locals()})

    pitch_list.append(pitch_seq[0])

    instrument_name = data_file_name.split('-')[0]
    start_pitch = data_file_name.split('.')[0].split('-')[1]
    if instrument_name not in pitch_dict:
        pitch_dict[instrument_name] = {}
    pitch_dict[instrument_name][start_pitch] = pitch_seq

def main():
    dataLoader = SoundDataLoader(
        WAV_PATH, is_load_all_data_dict=False, 
        time_frame_len=4, 
    )
    data_num = len(dataLoader.f_list)
    print(f"There are {data_num} files. ")

    models = loadModels()

    for data_i in range(data_num):
        if data_i % 16 == 0:
            print(f'Process: {data_i} / {data_num}', format(data_i, '.1%'))
        wav_path = dataLoader.data_path + dataLoader.f_list[data_i]
        data_tuple = dataLoader.load_a_audio_spec_from_disk(wav_path)
        data = torch.stack([data_tuple], dim=0)
        data = norm_log2(data, k=LOG_K)
        for name, model in models:
            evalModel(
                name, model, data, dataLoader, 
                dataLoader.f_list[data_i], 
            )

if __name__ == "__main__":
    main()
