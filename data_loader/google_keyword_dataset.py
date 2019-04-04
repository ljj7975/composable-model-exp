import os
import librosa
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from utils import is_audio_file

class GoogleKeywordDataset(Dataset):
    def __init__(self, root, target_class=None, size_per_class=None,
                 transform=None, unknown=False, silence=False, seed=0):
        self.silence_keyword = "__SILENCE__"
        self.unknown_keyword = "__UNKNOWN__"
        self.valid_keywords = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "wow", "yes", "zero"]
        self.sample_rate = 16000
        self.cache_rate = 0.7
        np.random.seed(seed)

        self.data_dir = root
        self.transform = transform
        
        self.keyword_audios = []
        self.background_noises = []
        self.targets = np.array([])
        self.audio_cache = {}

        if target_class:
            assert len(target_class) > 0
            if type(target_class[0]) == str:
                self.target_class = target_class
            else:
                self.target_class = []
                for index in target_class:
                    self.target_class.append(self.valid_keywords[index])

        else:
            self.target_class = self.valid_keywords
        self.unknown = unknown

        if len(self.target_class) == len(self.valid_keywords):
            self.unknown = False    

        if not self.target_class:
            self.target_class = list(np.arange(100))

        if len(self.target_class) < len(self.valid_keywords) and unknown:
            self.unknown = True

        self.class_to_idx = {}
        self.classes = []

        self.silence = silence
        if self.silence:
            self.class_to_idx[self.silence_keyword] = 0
            self.classes.append(self.silence_keyword)
        
        if self.unknown:
            self.class_to_idx[self.unknown_keyword] = len(self.classes)
            self.classes.append(self.unknown_keyword)
            
        class_index_offset = len(self.classes)
        for index, keyword in enumerate(self.target_class):
            self.class_to_idx[keyword] = index + class_index_offset
            self.classes.append(keyword)

        unknowns = []
        data_size = [0] * len(self.classes)

        for folder_name in os.listdir(self.data_dir):
            path_name = os.path.join(self.data_dir, folder_name)
            if os.path.isfile(path_name):
                continue
            if folder_name == "_background_noise_":
                label = self.silence_keyword
            elif folder_name in self.target_class:
                label = folder_name
            else:
                label = self.unknown_keyword

            if label == self.silence_keyword:
                for file_name in os.listdir(path_name):
                    wav_name = os.path.join(path_name, file_name)
                    if is_audio_file(wav_name):
                        self.background_noises.append(librosa.core.load(wav_name, sr=self.sample_rate)[0])
            elif label == self.unknown_keyword:
                unknowns = [os.path.join(path_name, file_name) for file_name in os.listdir(path_name)]
            else:
                class_index = self.class_to_idx[label]
                audios = [os.path.join(path_name, file_name) for file_name in os.listdir(path_name)]
                np.random.shuffle(audios)
                labels = np.zeros(len(audios)) + class_index

                data_size[class_index] = len(audios)
                self.keyword_audios += audios
                self.targets = np.concatenate((self.targets, labels))

        assert len(self.background_noises) > 0

        assert len(self.keyword_audios) == len(self.targets)

        avg_counts_per_keywords = int(len(self.keyword_audios) / len(self.target_class))

        if self.silence:
            silences = [self.silence_keyword] * avg_counts_per_keywords
            silence_index = self.class_to_idx[self.silence_keyword]
            labels = np.zeros(avg_counts_per_keywords) + silence_index

            data_size[silence_index] = len(avg_counts_per_keywords)
            self.keyword_audios += silences
            self.targets = np.concatenate((self.targets, labels))

        if self.unknown:
            np.random.shuffle(unknowns)
            unknowns = unknowns[:avg_counts_per_keywords]
            unknown_index = self.class_to_idx[self.unknown_keyword]
            labels = np.zeros(len(unknowns)) + unknown_index

            data_size[unknown_index] = len(unknowns)
            self.keyword_audios += unknowns
            self.targets = np.concatenate((self.targets, labels))

        print("< Dataset Summary >")
        print("\tseed\t:", seed)

        for keyword, label in self.class_to_idx.items():
            print("\t", keyword, "\t:", label, " (", data_size[label], ")")
        print("total data size : ", len(self.keyword_audios))

        self.targets = self.targets.astype(np.int_)


    def __getitem__(self, index):
        target = self.targets[index]

        file_name = self.keyword_audios[index]
        if file_name == self.silence_keyword:
            audio = np.zeros(self.sample_rate, dtype=np.float32)
        else:
            if random.random() < self.cache_rate and file_name in self.audio_cache:
                return self._transform(self.audio_cache[file_name]), target

            audio = librosa.core.load(self.keyword_audios[index], sr=self.sample_rate)[0]
            audio = np.pad(audio, (0, max(0, self.sample_rate - len(audio))), "constant")

        # add noise
        data = random.choice(self.background_noises)
        start_index = random.randint(0, len(data) - self.sample_rate - 1)
        data = data[start_index:start_index + self.sample_rate]
        data = (random.random() * 0.1) * data

        data += audio
        self.audio_cache[file_name] = data

        return self._transform(data), target

    def _transform(self, data):
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.targets)