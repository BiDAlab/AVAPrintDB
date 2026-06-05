from torch.utils.data import Dataset
from utils import VideoLoader
import pandas as pd
from tqdm import tqdm
import sys
import re
import os

class VideoDatasetAll(Dataset):
    def __init__(self, root_dir, generators=["GAGAvatar"], data=None, max_frames=None, device=None, transforms=None, split="dev", output_dir=None, reference_file=None, **kwargs):
        super().__init__()

        assert all(gen in ["GAGA", "HUNY", "LIVE"] for gen in generators)
        assert len(generators) > 0, "At least one generator must be provided"

        self.root_dir = root_dir
        self.generators = generators
        self.kwargs = kwargs
        self.max_frames = max_frames
        self.device = device
        self.transforms = transforms
        self.reference_file = reference_file
        
        if data is None:
            self.video_files_paths = []
            self.data = []

            for gen in generators:
                os.makedirs( os.path.join(output_dir,  split, gen), exist_ok=True )
                folder_path = os.path.join(root_dir, split, gen)
                print(f"Retrieving video files from directory {folder_path}")
                if not os.path.exists(folder_path):
                    continue
                videoPaths = os.listdir(folder_path)
                print(f"Found {len(videoPaths)} video files")

                # check csv file with names
                df = pd.read_csv( self.reference_file )
                df_subset = df[ df['generator'] == gen ]
                df_subset = df_subset[ df_subset['split'] == split ]
                videoPaths_csvs = df_subset[ "avatar_video_path" ]
                print(f"Found {len(videoPaths_csvs)} video files in csv")

                videoPaths = set(videoPaths).intersection( set( [ os.path.basename(x) for x in videoPaths_csvs] ) )
                print(f"Found {len(videoPaths)} after intersection with csv")


                for fname in tqdm(videoPaths, file=sys.stdout, desc="Loading dataset samples into memory - VideoDataset"):
                    videoname = os.path.join(folder_path, fname)
                    if not fname.endswith(".mp4"):
                        continue
                    if output_dir is not None:
                        outputFilename = os.path.join(output_dir,  split, gen,  fname.replace(".mp4", ".pt"))
                        if os.path.exists(outputFilename):
                            continue
                    self.data.append({'path': videoname, 'output_path': outputFilename})
                print(f"Found {len(self.data)} after filtering")
                    
        else:
            self.data = data
            self.video_files_paths =  [x["path"] for x in self.data]


        self.video_loader = VideoLoader(num_frames=self.max_frames, device=self.device, transform=self.transforms)


    def _extract_ids(self, filename):
        try:
            base = os.path.splitext(filename)[0]

            if re.match(r"\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", base):
                # Entire driver ID is RAVDESS
                parts = base.split("_", 1)
                driver_id = f"Actor_{parts[0][-2:]}"
                video_id = parts[0][:-3]
                target_id = parts[1]
            else:
                parts = base.split("_", 2)
                # General case: <DRIVER>_<VIDEO>_<TARGET>
                driver_id = parts[0]
                video_id = parts[1]
                target_id = parts[2]
            
            return driver_id, video_id, target_id
        except Exception as e:
            return None, None, None

    def _id_in_selected_datasets(self, id_str):
        if re.match(r"Actor_\d{2}", id_str):  # RAVDESS target
            return "RAVDESS" in self.datasets
        elif re.match(r"\d{4}", id_str):  # CREMA-D
            return "CREMA-D" in self.datasets
        return False

    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, idx):
        sample = self.data[idx]
        tensor = self.video_loader.load_video(sample["path"])
        return tensor, sample["output_path"]
    
    def collate_fn(self, batch):
        all_tensors = []
        all_paths = []

        for tensor, path in batch:
            all_tensors.append(tensor)
            all_paths.append(path)

        return all_tensors, all_paths
