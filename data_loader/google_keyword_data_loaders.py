import librosa
import numpy as np
from .google_keyword_dataset import GoogleKeywordDataset
from base import BaseDataLoader

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=4000, f_min=20, n_fft=480, hop_ms=10):
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms

    def compute_mfccs(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").astype(np.float32)
        return data

class GoogleKeywordDataLoader(BaseDataLoader):
    """
    loading Google Keyword data using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, size_per_class=None, training=True, target_class=None, unknown=True, seed=0):

        self.audioProcessor = AudioPreprocessor()

        self.data_dir = data_dir
        self.target_class = target_class
        self.unknown = unknown

        self.dataset = GoogleKeywordDataset(
                    self.data_dir,
                    target_class=target_class,
                    size_per_class=size_per_class,
                    transform=self.audioProcessor.compute_mfccs,
                    unknown=unknown,
                    seed=seed)

        super(GoogleKeywordDataLoader, self).__init__(
                    self.dataset,
                    batch_size, shuffle,
                    validation_split,
                    num_workers,
                    seed)