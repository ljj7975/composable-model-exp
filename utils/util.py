import errno
import hashlib
import os
from tqdm import tqdm
from utils import color_print as cp

def is_audio_file(file_name):
    return file_name.split('.')[-1] == "wav"

def get_instance(module, name, config, *args):
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True

def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str): Name to save the file under. If None, use the basename of the URL
        md5 (str): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )

def print_setting(data_loader, valid_data_loader, model, loss_fn, metrics, optimizer, lr_scheduler):
    if data_loader: cp.print_bold('TRAIN DATASET\n', data_loader, 'size :', len(data_loader.dataset))

    if valid_data_loader: cp.print_bold('VALID DATASET\n', valid_data_loader, 'size :', len(valid_data_loader.dataset))

    # if model: cp.print_bold('MODEL\n', model)

    if loss_fn: cp.print_bold('LOSS FUNCTION\n', loss_fn.__name__)

    if metrics: cp.print_bold('METRICS\n', [metric.__name__ for metric in metrics])

    if optimizer: cp.print_bold('OPTIMIZER\n', optimizer)

    if lr_scheduler: cp.print_bold('LR_SCHEDULER\n', type(lr_scheduler).__name__)
