import os
import torch
import numpy as np
import sys
from tqdm import tqdm
from cachetools import LRUCache
from decord import VideoReader, cpu, gpu

class VideoLoader:
    def __init__(self, max_videos=100000, num_frames=16, transform=None, device="cpu:0", **kwargs):
        self.cache = LRUCache(maxsize=max_videos)
        self.num_frames = num_frames
        self.transform = transform
        self.device = device
        self.context = self._resolve_context(device)


    def _resolve_context(self, device):
        if device is None:
            return cpu(0)
        if isinstance(device, torch.device):
            device_str = str(device)
        elif isinstance(device, str):
            device_str = device
        else:
        
            return cpu(0)

        if "cuda" in device_str:
            try:
                index = int(device_str.split(":")[1]) if ":" in device_str else 0
                return gpu(index)
            except Exception as e:
            
                return cpu(0)
        else:
            return cpu(0)


    def load_video(self, path):
        if path in self.cache:
            return self.cache[path]

        vr = VideoReader(path, ctx=self.context)
        total_frames = vr._num_frame  # This doesn't decode all frames

        if self.num_frames and total_frames >= self.num_frames:
            max_start = total_frames - self.num_frames
            start_idx = np.random.randint(0, max_start + 1)
            indices = np.arange(start_idx, start_idx + self.num_frames)
        else:
            indices = np.arange(total_frames, dtype=int)

        # Fetch only the required frames
        frames = vr.get_batch(indices).asnumpy()

        if self.transform is not None:
            frames = torch.stack([self.transform(frame) for frame in frames])
        else:
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0    

        return frames


    def load_videos(self, paths, verbose=False):
        video_list = []
        iterator = tqdm(paths, desc="Loading videos", total=len(paths), file=sys.stdout) if verbose else paths

        for path in iterator:
            video = self.load_video(path)
            video_list.append(video)

        return video_list

    @staticmethod
    def load_and_process(path, num_frames, device, use_gpu_ctx=True):
        try:
            context = gpu(int(device.split(":")[1])) if (device and "cuda" in device and use_gpu_ctx) else cpu()
            vr = VideoReader(path, ctx=context)
            total_frames = len(vr)
            indices = np.arange(total_frames)

            if num_frames and total_frames >= num_frames:
                max_start = total_frames - num_frames
                start_idx = np.random.randint(0, max_start + 1)
                indices = np.arange(start_idx, start_idx + num_frames)

            frames = vr.get_batch(indices).asnumpy()

            if frames is None or len(frames) == 0:
                raise ValueError(f"No frames loaded from {path}.")

            return (path, frames)

        except Exception as e:
            return (path, e)
        

