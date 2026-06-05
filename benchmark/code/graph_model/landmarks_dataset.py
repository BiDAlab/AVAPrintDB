import os
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from torch.utils.data import Dataset


from avatar_authentication.constants import _RAVDESS_RE, _CREMA_RE, _CREMAD_RAVDESS_RE, CREMAD_AND_RAVDESS_DATASETS, RAVDESS_DATASET, CREMAD_DATASET, FILENAME_SEPARATOR, ALLOWED_GENERATORS

class TripletGraphDataset(Dataset):
    """Dataset class for triplet graphs with landmarks data."""

    def __init__(self, 
        data_root:str, 
        dataset:str, 
        data_filter_csv:str,
        delaunay_edges_csv_path:str, 
        triplets_per_anchor:int=5, 
        id_list:list=[], 
        validation:bool=False, 
        num_frames = 50,
        frame_sampler: str = "random",   # "first" | "random" (extensible)
        pad_if_short: bool = False,
    ):
        """Initialize the dataset. It stores all the sample paths in memory, and they get loaded on-the-fly in `__getitem__`.

        Args:
            data_root (str): Path to directory where the landmark files are stored. They can be stored in subdirectories.
            dataset (str): Name of the dataset being used.
            data_filter_csv (str): if not None, only avatar videos in column avatar_video_path of this csv are included in the dataset.
            delaunay_edges_csv_path (str): path to csv file with delaunay edges definition
            triplets_per_anchor (int, optional): For each anchor, number of random triplets to generate. Defaults to 5.
            id_list (list, optional): Used to split the dataset into training and validation split. This ids are excluded from the training set if validation=True, i.e, these are the validation ids when validation is performed. Defaults to [].
            validation (bool, optional): Flag to indicate if the dataset instance is training or validation. Defaults to False.
            num_frames: Number of consecutive frames to return from each npy.
            frame_sampler: Strategy to pick consecutive frames:
                           - "first": always take first num_frames frames
                           - "random": take num_frames starting at a random valid index
            pad_if_short: If True and sequence shorter than num_frames, pad by repeating last frame.
        """
        super().__init__()
        self.triplets_per_anchor = triplets_per_anchor
        self.samples = []
        self.driver_to_indices = {}
        self.dataset = dataset
        assert dataset in [RAVDESS_DATASET, CREMAD_DATASET, CREMAD_AND_RAVDESS_DATASETS], f"Dataset {dataset} not supported."
        self.dataset_matcher = _RAVDESS_RE if dataset == RAVDESS_DATASET else (_CREMA_RE if dataset == CREMAD_DATASET else _CREMAD_RAVDESS_RE)
        self.num_frames: int = num_frames
        self.frame_sampler = frame_sampler
        self.pad_if_short = pad_if_short

        # Validate sampler choice
        valid_samplers = {"first", "random"}
        if self.frame_sampler not in valid_samplers:
            raise ValueError(f"Unknown frame_sampler='{self.frame_sampler}'. Valid: {sorted(valid_samplers)}")

        if self.num_frames is None or self.num_frames <= 0:
            raise ValueError("num_frames must be a positive integer.")

        if data_filter_csv is not None:
            valid_filenames = pd.read_csv(data_filter_csv)['avatar_video_path'].apply(lambda x: os.path.basename(x).split(".")[0]).tolist()
            print("Number of valid filenames according to filter csv:", len(valid_filenames))
        else:
            valid_filenames = None  # no filtering, accept all files that match dataset regex


        skipped = 0
        skipped_examples = []
        ignored = 0

        for root, _, files in os.walk(data_root):
            for file in tqdm(sorted(files)):
                if not file.endswith('.npy'):
                    continue
                if valid_filenames and os.path.basename(file).split(".")[0] not in valid_filenames:
                    ignored += 1
                    continue

                parts = file.split(FILENAME_SEPARATOR)
                driver = parts[1]
                target = parts[0]

                if not self.dataset_matcher.match(driver) or not self.dataset_matcher.match(target):
                    continue

                # train/val split filtering
                if validation and id_list != []:
                    if driver not in id_list or target not in id_list:
                        continue
                else:
                    if driver in id_list or target in id_list:
                        continue

                path = os.path.join(root, file)

                #length check (skip if too short)
                try:
                    arr = np.load(path, mmap_mode="r")   # cheap-ish: doesn't read whole file into RAM
                    T = arr.shape[0]
                except Exception as e:
                    skipped += 1
                    if len(skipped_examples) < 10:
                        skipped_examples.append((path, f"load_error: {e}"))
                    continue

                if T < self.num_frames:
                    skipped += 1
                    if len(skipped_examples) < 10:
                        skipped_examples.append((path, f"T={T} < num_frames={self.num_frames}"))
                    continue

                # keep valid sample
                idx = len(self.samples)
                self.samples.append((driver, path, target))
                self.driver_to_indices.setdefault(driver, []).append(idx)

        print(f"[TripletGraphDataset] Kept {len(self.samples)} samples. Skipped {skipped} too-short/bad files.")
        if skipped_examples:
            print("[TripletGraphDataset] Examples skipped:")
            for p, why in skipped_examples:
                print("  ", why, "->", p)
        if ignored > 0:
            print(f"[TripletGraphDataset] Ignored {ignored} samples not in {data_filter_csv}.")
        
        df = pd.read_csv(delaunay_edges_csv_path)
        # print("read delaunay edges:", df.shape)
        edges = list(zip(df['i'], df['j'])) + [(j, i) for (i, j) in zip(df['i'], df['j'])]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.num_nodes = int(max(df['i'].max(), df['j'].max()) + 1)
        # print("SAMPLES = ", len(self.samples))

        first_np = np.load(self.samples[0][1]).astype(np.float32)
        _, self.num_nodes, self.D = first_np.shape
        self.drivers = list(self.driver_to_indices.keys())

        

        



    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        if self.triplets_per_anchor == 1:
            return self._return_single_triplet(idx)
        else:
            return self._return_k_triplets(idx)
    
    def _slice_frames(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply dataset-wide frame slicing policy to a single sample array.
        Input arr: (T, N, D)
        Output: (T', N, D) where T' is num_frames.
        """
        T = arr.shape[0]
        N = self.num_frames

        if T == N:
            return arr

        if T > N:
            if self.frame_sampler == "first":
                start = 0
            elif self.frame_sampler == "random":
                # inclusive start in [0, T-N]
                start = np.random.randint(0, T - N + 1)
            else:
                # Should not happen due to validation in __init__
                raise ValueError(f"Unknown frame_sampler='{self.frame_sampler}'")
            return arr[start:start + N]

        # T < N
        if not self.pad_if_short:
            raise ValueError(f"Sequence too short: got T={T}, need num_frames={N}")

        # Pad by repeating the last available frame
        pad_len = N - T
        last = arr[-1:]  # shape (1, Nnodes, D)
        pad = np.repeat(last, pad_len, axis=0)
        return np.concatenate([arr, pad], axis=0)
    
    def _return_single_triplet(self, idx:int):
        """
        Return a triplet (anchor, positive, negative) for triplet-loss training.

        All three samples are sliced using the same dataset-wide frame strategy,
        producing tensors of shape (num_frames, num_nodes, D) if num_frames is set,
        otherwise variable-length (T', num_nodes, D).
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"idx {idx} out of bounds for dataset of length {len(self)}")

        # How many attempts before we "change reference element" (anchor_idx) and restart
        max_inner_tries = 50       # X: cap for the while loops
        max_anchor_resets = 10     # how many times we allow changing anchor before giving up

        # original anchor derived from idx
        anchor_idx0 = idx #// self.triplets_per_anchor

        # Precompute eligible anchors that CAN have a positive (>=2 samples for driver)
        # (optional but recommended)
        eligible_anchor_indices = [
            i for i, (drv, _, _) in enumerate(self.samples)
            if len(self.driver_to_indices.get(drv, [])) >= 2
        ]
        if not eligible_anchor_indices:
            raise RuntimeError("No eligible anchors: every driver has <2 samples, cannot form positives.")

        # Also ensure negatives are possible (>=2 distinct drivers with at least one sample)
        if len(self.drivers) < 2:
            raise RuntimeError("No negatives possible: dataset contains <2 drivers.")

        # We'll retry by changing the anchor reference if selection fails too much
        anchor_idx = anchor_idx0
        for reset in range(max_anchor_resets + 1):
            anchor_driver, anchor_path, _ = self.samples[anchor_idx]

            # --- load anchor ---
            anchor_np = np.load(anchor_path).astype(np.float32)

            # --- pick positive with cap ---
            pos_candidates = self.driver_to_indices.get(anchor_driver, [])
            if len(pos_candidates) < 2:
                # cannot pick a different positive for this anchor driver -> reset anchor
                anchor_idx = int(np.random.choice(eligible_anchor_indices))
                continue

            pos_idx = anchor_idx
            ok = False
            for _ in range(max_inner_tries):
                pos_idx = int(np.random.choice(pos_candidates))
                if pos_idx != anchor_idx:
                    ok = True
                    break
            if not ok:
                # too many failed attempts -> reset anchor
                anchor_idx = int(np.random.choice(eligible_anchor_indices))
                continue

            positive_np = np.load(self.samples[pos_idx][1]).astype(np.float32)

            # --- pick negative with cap ---
            neg_driver = anchor_driver
            ok = False
            for _ in range(max_inner_tries):
                neg_driver = str(np.random.choice(self.drivers))
                if neg_driver != anchor_driver and len(self.driver_to_indices.get(neg_driver, [])) > 0:
                    ok = True
                    break
            if not ok:
                # too many failed attempts -> reset anchor
                anchor_idx = int(np.random.choice(eligible_anchor_indices))
                continue

            neg_idx = int(np.random.choice(self.driver_to_indices[neg_driver]))
            negative_np = np.load(self.samples[neg_idx][1]).astype(np.float32)

            # --- slice frames ---
            anchor_np = self._slice_frames(anchor_np).astype(np.float32)
            positive_np = self._slice_frames(positive_np).astype(np.float32)
            negative_np = self._slice_frames(negative_np).astype(np.float32)

            return (
                torch.from_numpy(anchor_np),
                torch.from_numpy(positive_np),
                torch.from_numpy(negative_np),
            )

        raise RuntimeError(
            f"Failed to sample a valid triplet after {max_anchor_resets} anchor resets "
            f"and {max_inner_tries} inner tries each."
        )
    
    def _return_k_triplets(self, idx:int):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"idx {idx} out of bounds for dataset of length {len(self.samples)}")

        K = self.triplets_per_anchor
        max_inner_tries = 50

        # Precompute eligible anchors that CAN have a positive (>=2 samples for driver)
        eligible_anchor_indices = [
            i for i, (drv, _, _) in enumerate(self.samples)
            if len(self.driver_to_indices.get(drv, [])) >= 2
        ]
        if not eligible_anchor_indices:
            raise RuntimeError("No eligible anchors: every driver has <2 samples, cannot form positives.")
        if len(self.drivers) < 2:
            raise RuntimeError("No negatives possible: dataset contains <2 drivers.")

        # If this anchor can't form positives, resample an eligible anchor
        anchor_driver, anchor_path, _ = self.samples[idx]
        if len(self.driver_to_indices.get(anchor_driver, [])) < 2:
            idx = int(np.random.choice(eligible_anchor_indices))
            anchor_driver, anchor_path, _ = self.samples[idx]

        # --- load & slice anchor once ---
        anchor_np = np.load(anchor_path).astype(np.float32)
        anchor_np = self._slice_frames(anchor_np).astype(np.float32)
        anchor_t = torch.from_numpy(anchor_np)  # (Nframes, Nnodes, D)

        pos_candidates = self.driver_to_indices[anchor_driver]

        anchors = []
        positives = []
        negatives = []

        for _ in range(K):
            # --- pick positive (different sample, same driver) ---
            pos_idx = idx
            ok = False
            for _ in range(max_inner_tries):
                pos_idx = int(np.random.choice(pos_candidates))
                if pos_idx != idx:
                    ok = True
                    break
            if not ok:
                raise RuntimeError("Failed to sample a positive after many tries.")

            positive_np = np.load(self.samples[pos_idx][1]).astype(np.float32)
            positive_np = self._slice_frames(positive_np).astype(np.float32)
            positive_t = torch.from_numpy(positive_np)

            # --- pick negative (different driver) ---
            neg_driver = anchor_driver
            ok = False
            for _ in range(max_inner_tries):
                neg_driver = str(np.random.choice(self.drivers))
                if neg_driver != anchor_driver and len(self.driver_to_indices.get(neg_driver, [])) > 0:
                    ok = True
                    break
            if not ok:
                raise RuntimeError("Failed to sample a negative driver after many tries.")

            neg_idx = int(np.random.choice(self.driver_to_indices[neg_driver]))
            negative_np = np.load(self.samples[neg_idx][1]).astype(np.float32)
            negative_np = self._slice_frames(negative_np).astype(np.float32)
            negative_t = torch.from_numpy(negative_np)

            anchors.append(anchor_t)       # same anchor repeated
            positives.append(positive_t)
            negatives.append(negative_t)

        return (
            torch.stack(anchors, dim=0),   # (K, Nframes, Nnodes, D)
            torch.stack(positives, dim=0), # (K, Nframes, Nnodes, D)
            torch.stack(negatives, dim=0), # (K, Nframes, Nnodes, D)
        )



    
def collate_fn(batch):
    as_, ps, ns = zip(*batch)
    return torch.stack(as_, dim=0), torch.stack(ps, dim=0), torch.stack(ns, dim=0)

def collate_k(batch):
    a, p, n = zip(*batch)  # each is (K, F, N, D)
    a = torch.cat(a, dim=0)  # (B*K, F, N, D)
    p = torch.cat(p, dim=0)
    n = torch.cat(n, dim=0)
    return a, p, n

    

class TestingDataset(Dataset):
    def __init__(self, root_dir: str, dataset: str, data_filter_csv:str, num_frames = 50, frame_sampler: str = "random", pad_if_short: bool = False, generators=["GAGA"], slide_window_stride:int = 1):
        super().__init__()
        self.root_dir = root_dir
        self.dataset = dataset
        assert dataset in [RAVDESS_DATASET, CREMAD_DATASET, CREMAD_AND_RAVDESS_DATASETS], f"Dataset {dataset} not supported."
        self.dataset_matcher = _RAVDESS_RE if dataset == RAVDESS_DATASET else (_CREMA_RE if dataset == CREMAD_DATASET else _CREMAD_RAVDESS_RE)
        self.num_frames: int = num_frames
        self.frame_sampler = frame_sampler
        self.pad_if_short = pad_if_short
        self.generators = generators # a list, choices: [GAGA, LIVE, HUNY, FANP, NEMO, ORIG]
        self.slide_window_stride = slide_window_stride

        self.pairs = []

        # Validate sampler choice
        valid_samplers = {"first", "random", "sliding"}
        if self.frame_sampler not in valid_samplers:
            raise ValueError(f"Unknown frame_sampler='{self.frame_sampler}'. Valid: {sorted(valid_samplers)}")

        if self.num_frames is None or self.num_frames <= 0:
            raise ValueError("num_frames must be a positive integer.")
        
        if not isinstance(self.slide_window_stride, int) or self.slide_window_stride < 1:
            raise ValueError("slide_window_stride must be an integer >= 1.")
        
        bad = [g for g in self.generators if g not in ALLOWED_GENERATORS]
        if bad:
            raise ValueError(f"Unknown generators: {bad}. Allowed: {sorted(ALLOWED_GENERATORS)}")

        # Build valid_filenames list from root_dir, filtered by dataset regex
        # We treat files as .npy samples (since __getitem__ uses np.load)
        self.stem_to_path = self._collect_valid_files_by_stem(self.root_dir)

        print(f"Found {len(self.stem_to_path)} valid .npy files in {root_dir} matching dataset regex.")
        
        self._num_frames_cache = {}

        # Read CSV with columns enrolment_sample, test_sample, label
        df = pd.read_csv(data_filter_csv)
        df = self._expand_generators_df(df, self.generators)


        dropped = 0
        missing = 0
        too_short = 0

        for _, row in tqdm(df.iterrows()):
            enrol_name = self._to_stem(str(row["enrolment_sample"]))
            test_name = self._to_stem(str(row["test_sample"]))

            try:
                label = int(row["label"])
            except Exception:
                dropped += 1
                continue

            if label not in (0, 1):
                dropped += 1
                continue

            enrol_path = self.stem_to_path.get(enrol_name)
            test_path = self.stem_to_path.get(test_name)


            if enrol_path is None or test_path is None:
                missing += 1
                continue

            if self._get_T(enrol_path) < self.num_frames or self._get_T(test_path) < self.num_frames:
                too_short += 1
                continue

            self.pairs.append((enrol_path, test_path, label))

        if len(self.pairs) == 0:
            raise RuntimeError(
                "No valid pairs were loaded. Check that:\n"
                "- root_dir contains your .npy files\n"
                "- CSV enrol/test values match file stems (filename without extension)\n"
                "- dataset regex matches your filenames\n"
            )
    

    def _sliding_windows(self, arr: np.ndarray) -> np.ndarray:
        """
        Create sliding windows of length self.num_frames with stride self.slide_window_stride.
        Input arr: (T, Nnodes, D)
        Output: (W, num_frames, Nnodes, D), where W is the number of windows.
        """
        T = arr.shape[0]
        N = self.num_frames
        S = self.slide_window_stride

        if T < N:
            if not self.pad_if_short:
                raise ValueError(f"Sequence too short: got T={T}, need num_frames={N}")
            # Pad by repeating last frame up to N, then return a single window
            pad_len = N - T
            last = arr[-1:]  # (1, Nnodes, D)
            pad = np.repeat(last, pad_len, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
            return arr[None, ...]  # (1, N, Nnodes, D)

        # T >= N
        # starts: 0, S, 2S, ... <= T-N
        starts = range(0, T - N + 1, S)
        windows = np.stack([arr[s:s + N] for s in starts], axis=0)
        return windows  # (W, N, Nnodes, D)

    def _expand_generators_df(self, df: pd.DataFrame, generators):
        """
        Given a df with columns enrolment_sample/test_sample that end with '--GAGA.mp4',
        replicate rows for each generator and replace the trailing generator token.

        If generators == ['GAGA'] -> returns df unchanged.
        """
        gens = list(dict.fromkeys(generators))  # dedupe, preserve order
        if len(gens) == 0:
            gens = ["GAGA"]  # default to GAGA if empty
        if len(gens) == 1 and gens[0] == "GAGA":
            return df

        # Replace ONLY the final '--<GEN>.mp4' suffix (GEN can be any uppercase letters)
        # Example: "...--GAGA.mp4" -> "...--LIVE.mp4"
        suffix_re = re.compile(r"--[A-Z]+\.mp4$")

        def swap_gen(filename: str, new_gen: str) -> str:
            s = str(filename).strip()
            if suffix_re.search(s):
                return suffix_re.sub(f"--{new_gen}.mp4", s)
            # If the file doesn't match pattern, you can either:
            # - return unchanged
            # - or raise an error (safer)
            return s

        dfs = []
        for g in gens:
            tmp = df.copy()
            tmp["enrolment_sample"] = tmp["enrolment_sample"].map(lambda x: swap_gen(x, g))
            tmp["test_sample"]      = tmp["test_sample"].map(lambda x: swap_gen(x, g))
            dfs.append(tmp)

        return pd.concat(dfs, ignore_index=True)
    
    
    def _npy_shape(self, path: str):
        with open(path, "rb") as f:
            version = read_magic(f)
            if version == (1, 0):
                shape, _, _ = read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, _ = read_array_header_2_0(f)
            else:
                # Fallback: loads array (rarely needed)
                f.seek(0)
                shape = np.load(f).shape
        return shape

    def _get_T(self, npy_path: str) -> int:
        T = self._num_frames_cache.get(npy_path)
        if T is not None:
            return T
        T = int(self._npy_shape(npy_path)[0])
        self._num_frames_cache[npy_path] = T
        return T

    
    
    def _to_stem(self, s: str) -> str:
        """
        CSV values are filenames (maybe with or without extension).
        Return the filename *stem* (no directories, no extension).
        Examples:
          "abc.npy" -> "abc"
          "subdir/abc.npy" -> "abc"
          "abc" -> "abc"
        """
        s = s.strip()
        base = os.path.basename(s)
        stem, _ = os.path.splitext(base)
        return stem

    def _collect_valid_files_by_stem(self, root_dir: str):
        """
        Walk root_dir, collect .npy files that match dataset regex.
        Return dict: stem -> full_path

        If duplicate stems exist, we raise (better than silently picking one).

        Generates the 
        """
        stem_to_path = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if not fn.endswith(".npy"):
                    continue

                parts = fn.split(FILENAME_SEPARATOR)
                driver = parts[1]
                target = parts[0]

                if not self.dataset_matcher.match(driver) or not self.dataset_matcher.match(target):
                    continue

                full_path = os.path.normpath(os.path.abspath(os.path.join(dirpath, fn)))
                stem = os.path.splitext(fn)[0]

                if stem in stem_to_path and stem_to_path[stem] != full_path:
                    raise RuntimeError(
                        f"Duplicate stem '{stem}' found:\n"
                        f" - {stem_to_path[stem]}\n"
                        f" - {full_path}\n"
                        "Stems must be unique if CSV references only stems."
                    )

                stem_to_path[stem] = full_path

        return stem_to_path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"idx {idx} out of bounds for dataset of length {len(self)}")
        enrol_path, test_path, label = self.pairs[idx]

        with open(enrol_path, "rb") as f:
            arr_enrol = np.load(f).astype(np.float32, copy=False)

        enrol_windows = self._sliding_windows(arr_enrol).astype(np.float32, copy=False) 

        with open(test_path, "rb") as f:
            arr_test = np.load(f).astype(np.float32, copy=False)
        test_windows = self._sliding_windows(arr_test).astype(np.float32, copy=False)


        enrol_t = torch.from_numpy(enrol_windows)  # (E, N, Nnodes, D)
        test_t = torch.from_numpy(test_windows)    # (T, N, Nnodes, D)

        return enrol_t, test_t, torch.tensor(float(label), dtype=torch.float32), torch.tensor(idx, dtype=torch.long), enrol_path, test_path
    
    @staticmethod
    def collate_fn(batch):
        enrol_ws, test_ws, labels, idxs, enrol_paths, test_paths = zip(*batch)
        return list(enrol_ws), list(test_ws), torch.stack(labels), torch.stack(idxs), list(enrol_paths), list(test_paths)



if __name__ == "__main__":
    dataset = TripletGraphDataset(
        data_root="/mnt/data1/BIDALAB_AVATAR_DATABASE/109LANDMARKS/TEST/GAGA",
        dataset=CREMAD_DATASET,
        data_filter_csv="/home/laurapedrouzo/LAPR/BIDALAB_DATABASE_AVATAR/db_csvs/CREMA-D_all_with_avatar_paths.csv",
        delaunay_edges_csv_path="/mnt/data1/BIDALAB_AVATAR_DATABASE/109LANDMARKS/delaunay_edges.csv",
        triplets_per_anchor=5,
        id_list=["C1003", "C1004", "C1005"],
        validation=False,
        num_frames=30,
        frame_sampler="random",
        pad_if_short=False,
    )
    
    sample = dataset[0]
    print(f"Sample 0: {sample[0].shape, sample[1].shape, sample[2].shape}")