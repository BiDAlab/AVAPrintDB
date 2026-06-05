import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from avatar_authentication.constants import RAVDESS_DATASET, CREMAD_DATASET, CREMAD_AND_RAVDESS_DATASETS

_RAVDESS_RE = re.compile(r"^Actor_\d{2}$")   # e.g., Actor_01
_CREMA_RE   = re.compile(r"^C\d{4}$")         # e.g., C1022


class TripletGraphDataset(Dataset):
    """Dataset class that provides triplets for training with triplet loss.
    """
    def __init__(self, data_root:str, delaunay_edges_csv_path:str, triplets_per_anchor:int=5, id_list:list=[], validation:bool=False, originals:bool=False):
        """Initialize the dataset. It stores all the sample paths in memory, and they get loaded on-the-fly in `__getitem__`.

        Args:
            data_root (str): Path to directory where the preprocessed driver folders are located
            delaunay_edges_csv_path (str): path to csv file with delaunay edges definition
            triplets_per_anchor (int, optional): For each anchor, number of random triplets to generate. Defaults to 5.
            id_list (list, optional): Used to split the dataset into training and validation split. This ids are excluded from the training set if validation=True, i.e, these are the validation ids when validation is performed. Defaults to [].
            validation (bool, optional): Flag to indicate if the dataset instance is training or validation. Defaults to False.
            originals (bool, optional): Flag to indicate if the expected preprocessed data belongs to avatar videos or original videos (since the data structure changes). Defaults to False.
        """
        super().__init__()
        self.triplets_per_anchor = triplets_per_anchor
        self.samples = []
        self.driver_to_indices = {}

        if originals:
            # When using original videos, there is no "target" folder, since the driver and target are the same
            for driver in sorted(os.listdir(data_root)):
                if validation:
                    if driver not in id_list: continue
                else:
                    if driver in id_list: continue
                driver_dir = os.path.join(data_root, driver)
                if not os.path.isdir(driver_dir): continue
                for video in sorted(os.listdir(driver_dir)):
                        if video.startswith("02"): # These are videos with audio, we are not using them
                            continue
                        video_dir = os.path.join(driver_dir, video)
                        if not os.path.isdir(video_dir): continue
                        for window_file in sorted(os.listdir(video_dir)):
                            if not window_file.endswith('.npy'): continue
                            path = os.path.join(video_dir, window_file)
                            idx = len(self.samples)
                            self.samples.append((driver, path, driver)) # video original, mismo driver y target
                            self.driver_to_indices.setdefault(driver, []).append(idx)
        else:
            # For any avatar video this is the structure: driver/target/video/windows.npy
            for driver in sorted(os.listdir(data_root)):
                # print("Driver ", driver)
                if validation:
                    if driver not in id_list: continue
                else:
                    if driver in id_list: continue
                driver_dir = os.path.join(data_root, driver)
                if not os.path.isdir(driver_dir): continue
                for target in sorted(os.listdir(driver_dir)):
                    # print("\tTarget ", target)
                    if validation:
                        if target not in id_list: continue
                    else:
                        if target in id_list: continue
                    target_dir = os.path.join(driver_dir, target)
                    if not os.path.isdir(target_dir): continue
                    for video in sorted(os.listdir(target_dir)):
                        if video.startswith("02"): # These are videos with audio, we are not using them
                            continue
                        # print("\t\tVideo ", video)
                        video_dir = os.path.join(target_dir, video)
                        if not os.path.isdir(video_dir): continue
                        for window_file in sorted(os.listdir(video_dir)):
                            # print("\t\t\tWindow ", window_file)
                            if not window_file.endswith('.npy'): continue
                            path = os.path.join(video_dir, window_file)
                            idx = len(self.samples)
                            self.samples.append((driver, path, target))
                            self.driver_to_indices.setdefault(driver, []).append(idx)
                            # print("\t\t\t\tADDED SAMPLE")

        df = pd.read_csv(delaunay_edges_csv_path)
        edges = list(zip(df['i'], df['j'])) + [(j, i) for (i, j) in zip(df['i'], df['j'])]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.num_nodes = int(max(df['i'].max(), df['j'].max()) + 1)
        # print("SAMPLES = ", len(self.samples))

        first_np = np.load(self.samples[0][1]).astype(np.float32)
        self.T, self.num_nodes, self.D = first_np.shape
        self.drivers = list(self.driver_to_indices.keys())

    def __len__(self):
        return len(self.samples) * self.triplets_per_anchor

    def __getitem__(self, idx):
        """
        Return a triplet (anchor, positive, negative) for triplet-loss training.

        The dataset is conceptually organized by *anchors* (each element in
        ``self.samples``). To increase sample diversity, the reported length
        of the dataset is ``len(self.samples) * self.triplets_per_anchor``.
        Any index ``idx`` is mapped to its corresponding anchor via
        ``anchor_idx = idx // self.triplets_per_anchor``.

        For a given anchor, this method:
        1) Loads the anchor window from disk.
        2) Samples a **positive** window from the same driver (identity) as
            the anchor but from a *different* index (and thus typically a
            different video/window). Any target is allowed.
        3) Samples a **negative** window from a *different* driver than the
            anchor; video and target are unconstrained.

        The three windows are converted to ``torch.FloatTensor`` after
        slicing away the first time step (``arr[1:]``) (this is due to the 
        original dataset, in which the first frame is static, does not belong 
        to the video and messes up with the facial dynamics), preserving the
        remaining temporal dimension and node/feature dimensions.

        Parameters
        ----------
        idx : int
            Global dataset index in ``[0, len(self))``. Multiple consecutive
            indices may correspond to the same anchor, depending on
            ``self.triplets_per_anchor``.

        Returns
        -------
        anchor : torch.FloatTensor
            Tensor of shape ``(T-1, N, D)`` corresponding to the anchor
            window, where ``T`` is the original temporal length, ``N`` is the
            number of graph nodes, and ``D`` is the feature dimension.
        positive : torch.FloatTensor
            Tensor of shape ``(T-1, N, D)`` sampled from the *same driver*
            as the anchor but a *different* sample (index).
        negative : torch.FloatTensor
            Tensor of shape ``(T-1, N, D)`` sampled from a *different
            driver* than the anchor.

        Raises
        ------
        IndexError
            If ``idx`` is out of bounds.
        KeyError
            If the anchor driver is missing from ``self.driver_to_indices``.
        FileNotFoundError / OSError
            If an expected ``.npy`` file cannot be loaded.
        """
        anchor_idx = idx // self.triplets_per_anchor
        anchor_driver, anchor_path, anchor_target = self.samples[anchor_idx]
        anchor_np = np.load(anchor_path).astype(np.float32)

        # Positive sample: same driver, different video, any target
        pos_candidates = self.driver_to_indices[anchor_driver]
        pos_idx = anchor_idx
        while pos_idx == anchor_idx:
            pos_idx = np.random.choice(pos_candidates)
        positive_np = np.load(self.samples[pos_idx][1]).astype(np.float32)

        # Negative sample: different driver, any video, any target
        neg_driver = anchor_driver
        while neg_driver == anchor_driver: 
            neg_driver = np.random.choice(self.drivers)
        negative_np = np.load(
            self.samples[np.random.choice(self.driver_to_indices[neg_driver])][1]
        ).astype(np.float32)

        return (
            torch.from_numpy(anchor_np[1:]),
            torch.from_numpy(positive_np[1:]),
            torch.from_numpy(negative_np[1:]),
        )

def collate_fn(batch):
    as_, ps, ns = zip(*batch)
    return torch.stack(as_, dim=0), torch.stack(ps, dim=0), torch.stack(ns, dim=0)



def _matches_ravdess(name: str) -> bool:
    return bool(_RAVDESS_RE.match(name))

def _matches_crema(name: str) -> bool:
    return bool(_CREMA_RE.match(name))


def _is_valid_name(name: str, selected_dataset: str) -> bool:
    """Return True if `name` is valid given the dataset selection."""
    if selected_dataset == RAVDESS_DATASET:
        return _matches_ravdess(name)
    if selected_dataset == CREMAD_DATASET:
        return _matches_crema(name)
    # CREMAD_AND_RAVDESS_DATASETS: accept either format
    return _matches_ravdess(name) or _matches_crema(name)

class TestingDataset(Dataset):
    """
    Dataset for evaluating driver-target video windows stored as `.npy` arrays.

    This class loads either **original videos** (driver only) or
    **driver-target avatar videos** from a structured directory and
    provides them as PyTorch tensors. Each item consists of the driver
    identity, the target identity, and the corresponding feature tensor.

    Directory structure depends on the ``originals`` flag:

    - If ``originals=True`` (original videos only):
      ```
      root/
        DriverID/
          VideoID/
            window.npy
      ```
      In this case, the driver and target IDs are the same.

    - If ``originals=False`` (avatarized videos):
      ```
      root/
        DriverID/
          TargetID/
            VideoID/
              window.npy
      ```

    In both cases, window files must be saved as ``.npy`` arrays with
    shape ``(T, N, D)``, where:
      * ``T`` is the number of timesteps,
      * ``N`` is the number of nodes (e.g. landmarks, graph nodes),
      * ``D`` is the feature dimension.

    The dataset can be restricted to specific identity formats depending
    on the ``selected_dataset`` argument:

      - ``"RAVDESS"`` → only IDs of the form ``Actor_XX`` are valid
      - ``"CREMA"``   → only IDs of the form ``XXXX`` (4 digits) are valid
      - ``"CREMAD_AND_RAVDESS"``     → both formats are accepted

    Parameters
    ----------
    root_dir : str
        Path to the root directory containing the dataset.
    originals : bool, optional
        If True, load original driver-only videos. If False, load
        driver-target avatarized videos. Default is False.
    selected_dataset : {"RAVDESS", "CREMA", "CREMAD_AND_RAVDESS"}, optional
        Identity format filter. Defaults to "CREMAD_AND_RAVDESS".

    Attributes
    ----------
    originals : bool
        Indicates whether original or avatar videos are loaded.
    data_root : str
        Root directory of the dataset.
    selected_dataset : str
        Dataset filter type.
    samples : list of tuple
        Each entry is ``(driver, path, target)``, where:
          - ``driver`` : str, the driver ID
          - ``path``   : str, path to the `.npy` file
          - ``target`` : str, the target ID (same as driver if originals)
    driver_to_indices : dict
        Mapping from driver ID → list of sample indices belonging to
        that driver.

    Methods
    -------
    __getitem__(index)
        Load a window from disk, cast to float32, convert to tensor, and
        return ``(driver, target, tensor)``.
    __len__()
        Return the number of samples.
    collate_fn(batch)
        Static method to merge a list of samples into a batch. Returns
        ``(drivers, targets, tensor_batch)`` with tensors stacked on the
        first dimension.

    Notes
    -----
    - The first timestep of each window is skipped: ``arr[1:]`` is used
      instead of the full array (due to the original datasets, that 
      contain one frame at the beginning of every video that does not
      match the rest of the video).
    """
    def __init__(self, root_dir, originals=False, selected_dataset=CREMAD_AND_RAVDESS_DATASETS):
        super().__init__()
        self.originals = originals
        self.data_root = root_dir
        self.selected_dataset = selected_dataset
        self.samples = []
        self.driver_to_indices = {}
        
        self.__load_data()


    def __load_data(self):
        if self.originals:
            for driver in sorted(os.listdir(self.data_root)): 

                driver_dir = os.path.join(self.data_root, driver)
                if not os.path.isdir(driver_dir): continue
                if not _is_valid_name(driver, self.selected_dataset): continue
                for video in sorted(os.listdir(driver_dir)):
                        video_dir = os.path.join(driver_dir, video)
                        if not os.path.isdir(video_dir): continue
                        for window_file in sorted(os.listdir(video_dir)):
                            if not window_file.endswith('.npy'): continue
                            path = os.path.join(video_dir, window_file)
                            idx = len(self.samples)
                            self.samples.append((driver, path, driver)) # video original, mismo driver y target
                            self.driver_to_indices.setdefault(driver, []).append(idx)
        else:
            for driver in sorted(os.listdir(self.data_root)):
                # print("Driver ", driver)

                driver_dir = os.path.join(self.data_root, driver)
                if not os.path.isdir(driver_dir): continue
                if not _is_valid_name(driver, self.selected_dataset): continue
                for target in sorted(os.listdir(driver_dir)):
                    # print("\tTarget ", target)

 
                    target_dir = os.path.join(driver_dir, target)
                    if not os.path.isdir(target_dir): continue
                    if not _is_valid_name(target, self.selected_dataset): continue
                    for video in sorted(os.listdir(target_dir)):
                        # print("\t\tVideo ", video)
                        video_dir = os.path.join(target_dir, video)
                        if not os.path.isdir(video_dir): continue
                        for window_file in sorted(os.listdir(video_dir)):
                            # print("\t\t\tWindow ", window_file)
                            if not window_file.endswith('.npy'): continue
                            path = os.path.join(video_dir, window_file)
                            idx = len(self.samples)
                            self.samples.append((driver, path, target))
                            self.driver_to_indices.setdefault(driver, []).append(idx)

    def __getitem__(self, index):
        ref_driver, ref_path, ref_target = self.samples[index]
        arr = np.load(ref_path).astype(np.float32)
        tensor = torch.from_numpy(arr[1:])
        return ref_driver, ref_target, tensor

    def __len__(self):
        return len(self.samples)
    
    @staticmethod
    def collate_fn(batch):
        ref_drivers, ref_targets, tensors = zip(*batch)
        tensors = torch.stack(tensors, dim=0)
        return list(ref_drivers), list(ref_targets), tensors

