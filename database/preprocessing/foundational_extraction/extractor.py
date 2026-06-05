import torch

from tqdm import tqdm
import sys
import os

from backbones import get_backbone_and_transforms
from dataset import VideoDatasetAll





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", type=str, help="Path to directory with videos to preprocess with CLIP", default="database/data/videos/")
    parser.add_argument("-o", "--output-dir", type=str, help="Path to directory where pt files will be stored", default="database/data/embeddings/")
    parser.add_argument("-f", "--file-reference", type=str, help="Path to reference CSV file", default="./database/data/metadata/avaprintdb_metadata.csv")
    parser.add_argument("-b", "--backbone", type=str, default="CLIP", choices=["CLIP", "DINO"], help="Backbone model to use for feature extraction")
    parser.add_argument("-g", "--generators", type=str, nargs="+", default=["GAGA", "LIVE", "HUNY"], help="List of generators to include in the dataset (e.g: -g GAGA LIVE HUNY)")
    parser.add_argument("-s", "--split", type=str, default="DEV", help="Dataset split to process (e.g., DEV, TEST, or EVAL)")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="Device to run the backbone model on (e.g., 'cuda:0' or 'cpu')")
    args = parser.parse_args()

    device = torch.device(args.device)
    reference_file = args.file_reference
    output_dir = args.output_dir + "/" + args.backbone + "Feats"

    type_exe = args.split
    generators = args.generators

    
    backbone, transforms, run_inference = get_backbone_and_transforms(args.backbone, device=device)
    dataset = VideoDatasetAll(
        root_dir=args.input_directory,
        datasets=["RAVDESS", "CREMA-D"],
        generators=generators,
        device=None,
        transforms=transforms,
        output_dir=output_dir,
        split=type_exe,
        reference_file = reference_file
    )


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=8,
        collate_fn=dataset.collate_fn,
    )


    pbar = tqdm(enumerate(dataloader), file=sys.stdout, desc="Saving CLIP embeddings, processing batches", total=len(dataloader))

    with torch.no_grad():
        for i, batch in pbar:
            all_tensors, all_paths = batch

            for input_tensor, save_path in zip(all_tensors, all_paths):
                if os.path.exists(save_path):
                    continue

                input_tensor = input_tensor.unsqueeze(0).to(device)  # [1, T, C, H, W]
                if input_tensor.shape[1] > 512:  # if T > 100
                    feats = []
                    for j in range(0, input_tensor.shape[1], 512):
                        input_minibatch = input_tensor[:, j:j+512, :, :, :]
                        farl_feats = run_inference(backbone, input_minibatch)
                        feats.append(farl_feats)
                    farl_feats = torch.cat(feats, dim=1)  # [1, T, D]
                else:
                    farl_feats = run_inference(backbone, input_tensor)  # [1, T, D]
                valid_feats = farl_feats.squeeze(0)
                valid_feats = valid_feats.to(torch.float32)
                torch.save(valid_feats.cpu(), save_path)