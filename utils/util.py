import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def is_audio_file(file_name):
    return file_name.split('.')[-1] == "wav"

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])
