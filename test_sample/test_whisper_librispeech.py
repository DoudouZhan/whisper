"""
This script is actually inspired by notebooks/LibriSpeech.ipynb
"""

import os
import sys
import torch
import jiwer

import argparse
import whisper
import torchaudio
import numpy as np
import pandas as pd

from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    # Training settings
    parser = argparse.ArgumentParser(description='test whisper with librispeech')
    parser.add_argument('--librispeech-root', type=str, default=None,
                        help='root of librispeech dataset')
    parser.add_argument('--librispeech-subset', choices=['dev-clean', 'test-clean','test-0','test-10','test-20','test-30','test-40','test-50','test-60','test-70','test-80','test-90','test-100','test-200','test-300','test-400','test-500','test-600'], default='test-clean',
                        help='subset of librispeech dataset for evaluation')
    parser.add_argument('--whisper-model', choices=['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'], default='base',
                        help='whisper model to use')
    parser.add_argument('--whisper-weights-root', type=str, default=None,
                        help='whisper model to use')
    args = parser.parse_args()

    return args

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE, root=None):
        if root is None:
            root_path = os.path.expanduser("~/.cache")
            download = True
        else:
            root_path = root
            if os.path.isdir(os.path.join(root, "LibriSpeech/"+split)):
                download = False
            else:
                download = True
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root_path,
            url=split,
            download=download,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)




if __name__=="__main__":

    args = get_args()

    # create the dataloader
    dataset = LibriSpeech(args.librispeech_subset, root=args.librispeech_root)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16)

    # instantiate model 
    # use base model
    if args.whisper_weights_root is not None:
        # download the weights to the specified folder
        model = whisper.load_model(args.whisper_model, download_root=args.whisper_weights_root)
    else:
        model = whisper.load_model(args.whisper_model)

    # set decoding options
    # predict without timestamps for short-form transcription
    if DEVICE == 'cpu':
        options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
    else:
        options = whisper.DecodingOptions(language="en", without_timestamps=True)

    hypotheses = []
    references = []

    # decoding all sentences
    for mels, texts in tqdm(loader):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    # put into pandas dataframe
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))

    # text normalize
    normalizer = EnglishTextNormalizer()

    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    
    # calculate the word error rate (wer)
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))

    print(f"WER: {wer * 100:.2f} %")