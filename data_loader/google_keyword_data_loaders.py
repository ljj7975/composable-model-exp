import os
import librosa
from torch.utils.data.dataset import Dataset

from base import BaseDataLoader
from utils import is_audio_file

class GoogleKeywordDataset(Dataset):
    def __init__(self, keywords, data_dir):
        self.data_dir = data_dir
        self.audios = []
        self.labels = []
        self.background_noises = []

        label_mapping = {"silence":0, "unknown":1}

        for index, keyword in enumerate(keywords):
            label_mapping[keyword] = index+2

        for folder_name in os.listdir(data_dir):
            path_name = os.path.join(data_dir, folder_name)
            is_background_noise = False
            if os.path.isfile(path_name):
                continue
            if folder_name == "_background_noise_":
                is_background_noise = True
            elif folder_name in keywords:
                label = label_mapping[folder_name]
            else:
                label = label_mapping["unknown"]

            for file_name in os.listdir(path_name):
                wav_name = os.path.join(path_name, file_name)

                if is_background_noise and is_audio_file(file_name):
                    self.background_noises.append(path_name)
                else:
                    self.audios.append(wav_name)
                    self.labels.append(label)

        assert len(self.audios) == len(self.labels)

    def __getindex__(self, idx):
        return librosa.core.load(self.audios[idx], sr=16000)[0], self.labels[idx]

    def __len__(self):
        return len(self.labels)

class GoogleKeywordDataLoader(BaseDataLoader):
    """
    Google Keyword data loading using BaseDataLoader
    """
    def __init__(self, keywords, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        self.dataset = GoogleKeywordDataset(keywords, data_dir)
        super(GoogleKeywordDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    def __str__(self):
        return "Google Keyword Dataloader\n  total size : " + str(self.dataset.__len__())
