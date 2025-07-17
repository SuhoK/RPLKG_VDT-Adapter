import os
import pickle
import random
from scipy.io import loadmat
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class OxfordFlowers(DatasetBase):

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = root
        self.image_dir   = os.path.join(self.dataset_dir, "jpg")
        self.label_file  = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.setid_file  = os.path.join(self.dataset_dir, "setid.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")

        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        
        train, val, test = self.read_data()
      
        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)
    
    def read_data(self):
        #----------------------------#
        label_file = loadmat(self.label_file)["labels"][0]
        split_file = loadmat(os.path.join(self.dataset_dir, "setid.mat"))
        lab2cname = read_json(self.lab2cname_file)
        
        def _collate(indices):
            items = []
            for idx in indices:
                i = int(idx) - 1  # MATLAB index
                imname = f"image_{str(i+1).zfill(5)}.jpg"
                impath = os.path.join(self.image_dir, imname)
                if not os.path.isfile(impath):
                    print(f"[Missing file] {impath}")
                    continue
                label = int(label_file[i])
                cname = lab2cname[str(label)] #cname = lab2cname.get(str(label), "unknown")
                items.append(Datum(impath=impath, label=label - 1, classname=cname))
            return items
            
        train = _collate(split_file["trnid"][0])
        val   = _collate(split_file["valid"][0])
        test  = _collate(split_file["tstid"][0])
        self._classnames = [lab2cname[str(i)]                 
                            for i in sorted(lab2cname,        
                                            key=lambda x: int(x))]

        print(len(self.classnames))
        print(self._classnames[:10])
        return train, val, test

    @property
    def classnames(self):
        return self._classnames
