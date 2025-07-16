def read_data(self):
    def _collate(folder, label_offset=0):
        items = []
        lab2cname = read_json(self.lab2cname_file)
        classnames = sorted(os.listdir(folder))
        for class_idx, cname in enumerate(classnames):
            class_folder = os.path.join(folder, cname)
            if not os.path.isdir(class_folder):
                continue
            images = sorted(os.listdir(class_folder))
            for imname in images:
                impath = os.path.join(class_folder, imname)
                if os.path.isfile(impath):
                    items.append(Datum(impath=impath, label=class_idx + label_offset, classname=cname))
        return items

    # 전체 train 데이터
    trainval_items = _collate(os.path.join(self.dataset_dir, "train"))
    
    # 랜덤하게 80:20 비율로 train/val 나누기
    random.seed(42)  # reproducibility
    random.shuffle(trainval_items)
    n_total = len(trainval_items)
    n_val = int(n_total * 0.2)
    val = trainval_items[:n_val]
    train = trainval_items[n_val:]

    # test 데이터는 그대로 사용
    test = _collate(os.path.join(self.dataset_dir, "test"))

    return train, val, test
