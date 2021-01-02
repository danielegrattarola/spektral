import os
import os.path as osp
import zipfile

import requests
from tqdm import tqdm

DATASET_FOLDER = osp.expanduser("~/.spektral/datasets")


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
