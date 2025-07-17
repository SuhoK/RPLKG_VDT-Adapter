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
        self.dataset_dir = root #self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "jpg")
        self.label_file = os.path.join(self.dataset_dir, "imagelabels.mat")
        self.lab2cname_file = os.path.join(self.dataset_dir, "cat_to_name.json")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_OxfordFlowers.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)


        #if os.path.exists(self.split_path):
        #    train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        #else:
        #    train, val, test = self.read_data()
        #    OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
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
        
        # 파일명 생성 함수
        def get_imname(i):
            return f"image_{str(i).zfill(5)}.jpg"
        
        # 클래스명 로드
        lab2cname = read_json(self.lab2cname_file)

        #def _collate(folder, label_offset=0):
        def _collate(indices):
            items = []
            for idx in indices:
                i = int(idx) - 1  # MATLAB index는 1-based
                imname = get_imname(i + 1)
                impath = os.path.join(self.image_dir, imname)
                label = int(label_file[i])
                cname = lab2cname.get(str(label), "unknown")
                items.append(Datum(impath=impath, label=label - 1, classname=cname))
            return items
        train = _collate(split_file["trnid"][0])
        val = _collate(split_file["valid"][0])
        test = _collate(split_file["tstid"][0])

        return train, val, test
            #lab2cname = read_json(self.lab2cname_file)
            #classnames = sorted(os.listdir(folder))
            #for class_idx, cname in enumerate(classnames):
            #    class_folder = os.path.join(folder, cname)
            #    if not os.path.isdir(class_folder):
            #        continue
            #    images = sorted(os.listdir(class_folder))
            #    for imname in images:
            #        impath = os.path.join(class_folder, imname)
            #        if os.path.isfile(impath):
            #            items.append(Datum(impath=impath, label=class_idx + label_offset, classname=cname))
            #return items
    
        
        #trainval_items = _collate(os.path.join(self.dataset_dir, "train"))
        
        #random.seed(42)  # reproducibility
        #random.shuffle(trainval_items)
        #n_total = len(trainval_items)
        #n_val = int(n_total * 0.2)
        #val = trainval_items[:n_val]
        #train = trainval_items[n_val:]
        #test = _collate(os.path.join(self.dataset_dir, "test"))
    
        #return train, val, test
