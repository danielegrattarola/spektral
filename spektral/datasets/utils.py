import os
import os.path as osp
import zipfile

import requests
from tqdm import tqdm

_dataset_folder = os.path.join("~", "spektral", "datasets")
_config_path = osp.expanduser(os.path.join("~", "spektral", "config.json"))
if osp.isfile(_config_path):
    import json

    with open(_config_path) as fh:
        _config = json.load(fh)
        _dataset_folder = _config.get("dataset_folder", _dataset_folder)
DATASET_FOLDER = osp.expanduser(_dataset_folder)


def download_file(url, datadir, fname, progress=True, extract=True):
    with requests.get(url, stream=progress) as r:
        r.raise_for_status()
        os.makedirs(datadir, exist_ok=True)
        outfile = osp.join(datadir, fname)
        with open(outfile, "wb") as of:
            if progress:
                pbar = tqdm(
                    total=int(r.headers["Content-Length"]),
                    ncols=80,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk is not None:
                        of.write(chunk)
                        pbar.update(len(chunk))
            else:
                of.write(r.content)

    if extract and fname.endswith(".zip"):
        with zipfile.ZipFile(outfile, "r") as of:
            of.extractall(datadir)
        os.remove(outfile)
