import os
import os.path as osp
import shutil
from glob import glob

from joblib import Parallel, delayed
from tqdm import tqdm

from spektral.data import Dataset
from spektral.datasets.utils import download_file
from spektral.utils import load_off, one_hot


class ModelNet(Dataset):
    """
    The ModelNet10 and ModelNet40 CAD models datasets from the paper:

    > [3D ShapeNets: A Deep Representation for Volumetric Shapes](https://arxiv.org/abs/1406.5670)<br>
    > Zhirong Wu et al.

    Each graph represents a CAD model belonging to one of 10 (or 40) categories.

    The models are polygon meshes: the node attributes are the 3d coordinates
    of the vertices, and edges are computed from each face. Duplicate edges are
    ignored and the adjacency matrix is binary.

    The dataset are pre-split into training and test sets: the `test` flag
    controls which split is loaded.

    **Arguments**

    - `name`: name of the dataset to load ('10' or '40');
    - `test`: if True, load the test set instead of the training set.
    - `n_jobs`: number of CPU cores to use for reading the data (-1, to use all
    available cores)
    """

    url = {
        "10": "http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
        "40": "http://modelnet.cs.princeton.edu/ModelNet40.zip",
    }

    def __init__(self, name, test=False, n_jobs=-1, **kwargs):
        if name not in self.available_datasets():
            raise ValueError(
                "Unknown dataset {}. See {}.available_datasets() for a complete list of"
                "available datasets.".format(name, self.__class__.__name__)
            )
        self.name = name
        self.test = test
        self.n_jobs = n_jobs
        super().__init__(**kwargs)

    @property
    def path(self):
        return osp.join(super().path, self.name)

    def read(self):
        folders = glob(osp.join(self.path, "*", ""))
        dataset = "test" if self.test else "train"
        classes = [f.split("/")[-2] for f in folders]
        n_out = len(classes)

        print("Loading data")

        def load(fname, class_i):
            graph = load_off(fname)
            graph.y = one_hot(class_i, n_out)
            return graph

        output = []
        for i, c in enumerate(tqdm(classes)):
            fnames = osp.join(self.path, c, dataset, "{}_*.off".format(c))
            fnames = glob(fnames)
            output_partial = Parallel(n_jobs=self.n_jobs)(
                delayed(load)(fname, i) for fname in fnames
            )
            output.extend(output_partial)

        return output

    def download(self):
        print("Downloading ModelNet{} dataset.".format(self.name))
        url = self.url[self.name]
        download_file(url, self.path, "ModelNet" + self.name + ".zip")

        # Datasets are zipped in a folder: unpack them
        parent = self.path
        subfolder = osp.join(self.path, "ModelNet" + self.name)
        for filename in os.listdir(subfolder):
            shutil.move(osp.join(subfolder, filename), osp.join(parent, filename))
        os.rmdir(subfolder)
        shutil.rmtree(osp.join(self.path, "__MACOSX"), ignore_errors=True)

    @staticmethod
    def available_datasets():
        return ["10", "40"]
