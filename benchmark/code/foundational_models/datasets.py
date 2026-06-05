import os
import sys
import re
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import csv



class VideoDatasetTripletsFeatsSimple(Dataset):

    def __init__(self, root_dir, datasets=["CelebDF"], generators=["GAGAvatar"], anchors=None, data=None, max_frames=None, ktriplets=None, device=None, transforms=None, split="dev", validation_use=False,  **kwargs):
        super().__init__()
    

        # assert all(gen in ["GAGAvatar"] for gen in generators), "generators must be from ['GAGAvatar']"
        assert all(ds in ["RAVDESS", "CREMA-D"] for ds in datasets), "datasets must be from ['CelebDF', 'RAVDESS', 'CREMA-D']"
        assert len(datasets) > 0, "At least one dataset must be provided"
        assert len(generators) > 0, "At least one generator must be provided"

        self.root_dir = root_dir
        self.datasets = datasets
        self.generators = generators
        self.kwargs = kwargs
        self.max_frames = max_frames
        self.device = device
        self.transforms = transforms
        self.split = split
        self.validation_use = validation_use
        self.k = ktriplets

        if data is None:
            self.video_files_paths = []
            self.data = []

            for gen in generators:
                folder_path = os.path.join(root_dir, split, gen)
            
                print(f"Retrieving video files from directory {folder_path}")
                if not os.path.exists(folder_path):
                    continue
                for fname in tqdm(os.listdir(folder_path), file=sys.stdout, desc="Loading dataset samples into memory - VideoDatasetTripletsFeatsSimple"):
                
                    if not fname.endswith(".pt"):
                        continue
                    driver_id, video_id, target_id = self._extract_ids(fname)
                
                    if driver_id and target_id:
                        if self._id_in_selected_datasets(driver_id) and self._id_in_selected_datasets(target_id):
                        
                            self.video_files_paths.append(os.path.join(folder_path, fname))
                            self.data.append({
                                'path': os.path.join(folder_path, fname),
                                'driver_id': driver_id,
                                'video_id': video_id,
                                'target_id': target_id
                            })
                        else:
                            None             
        else:
            self.data = data
            self.video_files_paths =  [x["path"] for x in self.data]

        if anchors is not None:
            self.anchors = anchors
        else:
            self.anchors = self._get_anchors()
    
        paths = [ x['path'] for x  in self.data ]

        self.videos_dict = {}
        for one_path in paths:
            feat = self.load_feat(one_path)
            if feat is not None:
                self.videos_dict[one_path] = feat

        # Filtrar data
        self.data = [entry for entry in self.data if entry['path'] in self.videos_dict]

        # --- OPTIMIZACIÓN DE MEMORIA (Soluciona los 10ms de stack) ---
        self.ordered_paths = list(self.videos_dict.keys())
        # Creamos un solo bloque de memoria contiguo
        self.video_tensor = torch.stack([self.videos_dict[p] for p in self.ordered_paths])

        self.path_to_idx = {path: i for i, path in enumerate(self.ordered_paths)}
        del self.videos_dict # Liberamos el diccionario fragmentado

        # --- OPTIMIZACIÓN DE BÚSQUEDA (Soluciona los 18ms de lógica) ---
        self.driver_to_idx = defaultdict(list)
        self.target_to_idx = defaultdict(set)
        for i, entry in enumerate(self.data):
            self.driver_to_idx[entry['driver_id']].append(i)
            self.target_to_idx[entry['target_id']].add(i)

        self.all_indices_set = set(range(len(self.data)))

        # 1. Agrupamos los índices por Driver
        self.driver_to_idx = defaultdict(list)
        # 2. Agrupamos los índices por Target (como set para búsquedas rápidas)
        self.target_to_idx = defaultdict(set)

        for i, v in enumerate(self.data):
            self.driver_to_idx[v['driver_id']].append(i)
            self.target_to_idx[v['target_id']].add(i)

        # 3. Un set con todos los índices posibles para calcular negativos rápido
        self.all_indices_set = set(range(len(self.data)))
    
    def load_feat(self, path):
        import numpy as np
        vr = torch.load(path)
        total_frames = vr.shape[0]  # This doesn't decode all frames

        if self.max_frames and total_frames >= self.max_frames:
            max_start = total_frames - self.max_frames
            start_idx = np.random.randint(0, max_start + 1)
            indices = np.arange(start_idx, start_idx + self.max_frames)
        else:
            return None
            
        frames = vr[indices]
        return frames

    def _extract_ids(self, filename):
        base = os.path.splitext(filename)[0]
        target_id, driver_id, video_id, gen = base.split('--')
        return driver_id, video_id, target_id

    def _id_in_selected_datasets(self, id_str):
        if re.match(r"id\d{1,2}", id_str):  # CelebDF
            return "CelebDF" in self.datasets
        elif re.match(r"Actor_\d{2}", id_str):  # RAVDESS target
            return "RAVDESS" in self.datasets
        elif re.match(r"C\d{4}", id_str):  # CREMA-D
            return "CREMA-D" in self.datasets
        elif re.match(r"zz-zz-zz-zz-zz-zz-\d{2}", id_str):  # RAVDESS driver
            return "RAVDESS" in self.datasets
        return False

    def _get_anchors(self):
        self_reenactments = [entry for entry in self.data if entry["driver_id"]==entry["target_id"]]
    
        return self_reenactments
    
    def get_train_dataset_with_n_ids(self, num_ids=2, seed=42):
        """
        Generates a training dataset with a limited number of unique identities (drivers).
        This is used for experiments to see how performance scales with more identities.
        """
        random.seed(seed)

        train_id_set = []
        if "CREMA-D" in self.datasets:
        
            train_id_set = ['C1036', 'C1038', 'C1039', 'C1050', 'C1034', 'C1080', 'C1023', 'C1079', 'C1020', 'C1042', 'C1030', 'C1086', 'C1062', 'C1085', 'C1028', 'C1043', 'C1025', 'C1089', 'C1067', 'C1075', 'C1014', 'C1084', 'C1040', 'C1071', 'C1032', 'C1077', 'C1026', 'C1065', 'C1015', 'C1021', 'C1044', 'C1060', 'C1066', 'C1057', 'C1069', 'C1063', 'C1047', 'C1088', 'C1046', 'C1054', 'C1082', 'C1024', 'C1037', 'C1013', 'C1006', 'C1052', 'C1051', 'C1012', 'C1073', 'C1010', 'C1017', 'C1016', 'C1011', 'C1087', 'C1022', 'C1056', 'C1053', 'C1035', 'C1055', 'C1045', 'C1001']
        if "RAVDESS" in self.datasets:
        
            train_id_set = ['Actor_06', 'Actor_08', 'Actor_04', 'Actor_13', 'Actor_05', 'Actor_15', 'Actor_03', 'Actor_23', 'Actor_07', 'Actor_09', 'Actor_10', 'Actor_24', 'Actor_14', 'Actor_16', 'Actor_12', 'Actor_11']

        if train_id_set != -1:
            train_id_set = train_id_set[:num_ids]  
    
        
        selected_data = [entry for entry in self.data if entry['driver_id'] in train_id_set and entry['target_id'] in train_id_set]
        
        return VideoDatasetTripletsFeatsSimple(
            root_dir=self.root_dir,
            datasets=self.datasets,
            generators=self.generators,
            max_frames=self.max_frames,
            device=self.device,
            transforms=self.transforms,
            data=selected_data,
            split=self.split,
        )
        

    def get_train_val_dataset_split(self, val_size=0.2, seed=42):
        """
        Splits the full dataset such that no identity (driver_id or target_id)
        is shared between training and validation splits.
        """
    
        train_id_set = set()
        val_id_set = set()
        
        if "CREMA-D" in self.datasets:
        
            train_id_set = train_id_set.union({'C1047', 'C1023', 'C1015', 'C1038', 'C1057', 'C1020', 'C1080', 'C1075', 'C1012', 'C1089', 'C1067', 'C1022', 'C1084', 'C1045', 'C1051', 'C1088', 'C1065', 'C1021', 'C1042', 'C1079', 'C1013', 'C1010', 'C1087', 'C1032', 'C1001', 'C1006', 'C1011', 'C1062', 'C1060', 'C1050', 'C1014', 'C1044', 'C1037', 'C1028', 'C1055', 'C1063', 'C1043', 'C1036', 'C1071', 'C1035', 'C1024', 'C1052', 'C1054', 'C1077', 'C1039', 'C1073', 'C1017', 'C1034'})
            val_id_set = val_id_set.union({'C1066', 'C1085', 'C1016', 'C1053', 'C1025', 'C1056', 'C1086', 'C1069', 'C1030', 'C1040', 'C1082', 'C1026', 'C1046'})
        if "RAVDESS" in self.datasets:
        
            train_id_set = train_id_set.union({'Actor_05', 'Actor_09', 'Actor_16', 'Actor_10', 'Actor_13', 'Actor_15', 'Actor_23', 'Actor_03', 'Actor_08', 'Actor_11', 'Actor_24', 'Actor_07'})
            val_id_set = val_id_set.union({'Actor_14', 'Actor_06', 'Actor_04', 'Actor_12'})


        train_anchors = [a for a in self.anchors if a["driver_id"] in train_id_set and a["target_id"] in train_id_set]
        val_anchors = [a for a in self.anchors if a["driver_id"] in val_id_set and a["target_id"] in val_id_set]


        listTrainAnchors = np.unique( [ x['driver_id'] for x  in train_anchors ]).tolist()
        data_train = [ x for x in self.data if x['driver_id'] in listTrainAnchors and x['target_id'] in listTrainAnchors]
        train_dataset = VideoDatasetTripletsFeatsSimple(
            root_dir=self.root_dir,
            datasets=self.datasets,
            generators=self.generators,
            anchors=train_anchors,
            max_frames=self.max_frames,
            device=self.device,
            transforms=self.transforms,
            data=data_train,
            split=self.split,
        )

        listValidationAnchors = np.unique( [ x['driver_id'] for x  in val_anchors ]).tolist()
        data_val = [ x for x in self.data if x['driver_id'] in listValidationAnchors and x['target_id'] in listValidationAnchors]
        val_dataset = VideoDatasetTripletsFeatsSimple(
            root_dir=self.root_dir,
            datasets=self.datasets,
            generators=self.generators,
            anchors=val_anchors,
            max_frames=self.max_frames,
            device=self.device,
            transforms=self.transforms,
            data=data_val,
            # data=self.data,
            split=self.split,
            validation_use=True
        )

        return train_dataset, val_dataset
    
    def __len__(self):
        return len(self.data)
    
    def _get_triplet_2(self, idx):       
        self.k=6 
        anchor = self.data[idx]   
        # anchor = self.anchors[idx]

        anchor_driver_id = anchor['driver_id']
        anchor_target_id = anchor['target_id']

        all_positives = [v for v in self.data if v['driver_id'] == anchor_driver_id]

        self_reenactments = [v for v in all_positives if v['target_id'] == anchor_target_id ]
        replace_flag = len(self_reenactments) < self.k
        self_reenactments = np.random.choice(self_reenactments, self.k, replace=replace_flag).tolist()
        positive = self_reenactments

        if not self.validation_use:
            crossPull_reenactments = [v for v in all_positives if v['target_id'] != anchor_target_id ]
            replace_flag = len(crossPull_reenactments) < self.k
            crossPull_reenactments = np.random.choice(crossPull_reenactments, self.k, replace=replace_flag).tolist()
            positive = self_reenactments + crossPull_reenactments
        
        ##%%%%%

        all_negatives = [v for v in self.data if v['driver_id'] != anchor_driver_id ]

        self_reenactments = [v for v in all_negatives if v['target_id'] == anchor_target_id ]
        replace_flag = len(self_reenactments) < self.k
        self_reenactments = np.random.choice(self_reenactments, self.k, replace=replace_flag).tolist()
        negative = self_reenactments

        if not self.validation_use:
            crossPull_reenactments = [v for v in all_negatives if v['target_id'] != anchor_target_id ]
            replace_flag = len(crossPull_reenactments) < self.k
            crossPull_reenactments = np.random.choice(crossPull_reenactments, self.k, replace=replace_flag).tolist()
            negative = self_reenactments + crossPull_reenactments

        triplets = [anchor] + positive + negative
        labels = [0] + [1] * len(positive) + [-1] * len(negative)
        return triplets, labels
    

    def __getitem__(self, idx):

        triplets, labels = self._get_triplet_2(idx)

        # inputs = [ self.videos[x['path']] for x in triplets ] 
        indices = [self.path_to_idx[m['path']] for m in triplets]

        # inputs = torch.stack(inputs, dim=0)
        inputs = self.video_tensor[indices]
        return inputs, labels

    
    def collate_fn(self, batch):  # Stack the tensors along a new dimension
        inputs, labels = zip(*batch)  # Unzip the batch into separate lists
        inputs = torch.stack(inputs, dim=0)  # Stack the input tensors
        labels = torch.tensor(labels)  # Convert labels to a tensor
        return inputs, labels

if __name__ == "__main__":
    # Example usage
    datasize = []
    for max_frames in range(10,101, 10):
        dataset = VideoDatasetTripletsFeatsSimple(
            root_dir="/mnt/data2/CLIPFeats/",
            generators=["GAGA"],
            datasets=["RAVDESS"],
            split="DEV",
            max_frames=max_frames,
            ktriplets = 6,
            preload=True
        )
        # dataset = dataset.get_train_dataset_with_n_ids(num_ids=61)
        datasize.append((max_frames, len(dataset)))
    
    print(f"max_frames;dataset_size")
    for max_frames, size in datasize:
        print(f"{max_frames};{size}")


class EvalDatasetFeatsWindows(Dataset):


    def __init__(self, root_dir=None, evaluation_pairs_csv=None, max_frames=None, max_videos=1000, transforms=None, preload=True, **kwargs):
        super().__init__()
        assert root_dir is not None, "root_dir must be provided"
        assert evaluation_pairs_csv is not None, "Evaluation pairs CSV file path must be provided"
        assert max_frames is not None, "max_frames must be provided"
        assert os.path.exists(evaluation_pairs_csv), f"CSV file {evaluation_pairs_csv} does not exist"
        assert os.path.exists(root_dir), f"Directory root_dir: {root_dir} does not exist"

        self.transforms = transforms
        self.max_frames = max_frames
        self.max_videos = max_videos
        self.pairs = []  # (enrol_path, test_path, label)
        self.loaded_videos = {}
        self.root_dir = root_dir + os.path.basename(evaluation_pairs_csv).split('_')[-1][:-4] + '/' # This directory contains the mp4 files for evaluation
        self.preload = preload
        # Parse CSV and normalize paths
        enrol_paths = []
        test_paths = []
        with open(evaluation_pairs_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                enrol_path = os.path.join(self.root_dir, row['enrolment_sample'])
                test_path = os.path.join(self.root_dir, row['test_sample'])
                label = int(row['label'])

                self.pairs.append((enrol_path, test_path, label))
                enrol_paths.append(enrol_path)
                test_paths.append(test_path)

        unique_paths = set(enrol_paths + test_paths)
        invalidPaths = []

        if preload:

            windows_list = []
            video_to_range = {}
            current_idx = 0

            pbar = tqdm(unique_paths, total=len(unique_paths),
                        desc="Loading videos in cache...", file=sys.stdout)

            for path in pbar:
                try:
                    feat = self.load_feat(path.replace(".mp4", ".pt"))
                    # feat shape: [N_windows_video, F, D]
                    if feat is not None:
                        n_w = feat.shape[0]
                        windows_list.append(feat)
                        video_to_range[path] = (current_idx, current_idx + n_w)
                        current_idx += n_w
                    else:
                        invalidPaths.append(path)

                except Exception as e:
                
                    raise e

            self.windows_tensor = torch.cat(windows_list, dim=0).contiguous()
            self.video_to_range = video_to_range
            
        if invalidPaths:
            for path in tqdm(invalidPaths):
                self.pairs = [pair for pair in self.pairs if pair[0] != path and pair[1] != path]
    
    def load_feat(self, path):
        vr = torch.load(path)          # [T, D]
        total = vr.shape[0]
        if total < self.max_frames:
            return None

        w = self.max_frames
        step = w // 2
        windows = []
        for s in range(0, total - w + 1, step):
            windows.append(vr[s:s+w])  # slicing directo

        return torch.stack(windows, dim=0)  # [Nw, F, D]

    def __len__(self):
        return len(self.pairs)


    def __getitem__(self, idx):
        enrol_path, test_path, label = self.pairs[idx]

        e0, e1 = self.video_to_range[enrol_path]
        t0, t1 = self.video_to_range[test_path]

        return {
            "enrol_range": (e0, e1),
            "test_range": (t0, t1),
            "label": label,
            "enrol_path": enrol_path,
            "test_path": test_path,
        }

    @staticmethod
    def collate_fn(batch):
        enrol_ranges = torch.tensor([b["enrol_range"] for b in batch], dtype=torch.int64)  # [B,2]
        test_ranges  = torch.tensor([b["test_range"]  for b in batch], dtype=torch.int64)  # [B,2]
        label = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
        enrol_path = [b["enrol_path"] for b in batch]
        test_path  = [b["test_path"]  for b in batch]
        return enrol_ranges, test_ranges, label, enrol_path, test_path
    

if __name__ == "__main__":
    # Example usage
    datasize = []
    for max_frames in range(10,101, 10):
        dataset = EvalDatasetFeatsWindows(
            root_dir="/mnt/data2/CLIPFeats/TEST/",
            evaluation_pairs_csv="./benchmark/eval_files/evaluation_pairs_RAVDESS_GAGA.csv",
            max_frames=max_frames,
            preload=True
        )
        datasize.append((max_frames, len(dataset)))
    
    print(f"max_frames;dataset_size")
    for max_frames, size in datasize:
        print(f"{max_frames},{size}")