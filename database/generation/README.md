# AVAPrintDB Generation


The original videos used as driving videos and target identities can be downloaded from:

- RAVDESS: https://zenodo.org/records/1188976
- CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D


## Target Identities

In the folder `target_identities` we provide a reference image for each identity in those datasets to use as the target images for the avatars. We selected frontal poses with neutral expressions.




## CSV Files with Target-Driver Pairs


> [!IMPORTANT] 
> Make sure you edit the csv files in this folder so that the paths point to the correct location of the source datasets in your machine.

The csv files provided in this folder contain the target-driver combinations used to generate AVAPrintDB. They have the following columns:
- `driver_id`: String containing the identifier of the identity in the source video. For Ravdess videos it has the format of "Actor_XX", and for CREMA-D videos it has the format of "CXXXX", being XXXX the 4-digit zero-padded identifier assigned by CREMA-D authors.
- `driver_video_path`: absolute path to the location of the video. Make sure you edit these paths.
- `driver_video_randomname`: UUID generated for the source video, unique for each video.
- `target_id`: String containing the identifier of the identity of the target image used as the appearance to generate the avatar. 
- `target_image_path`: Path to the corresponding target image. Make sure you edit these paths.
- `split`: string containing "DEV" if the avatar generated from that row corresponds to the development split, or "TEST" if the avatar generated from that row corresponds to the evaluation split.

The selected cross-reenactment target identities were chosen to provide:

- demographic diversity
- balanced appearance variation

The exact identity combinations used in the official release are fully specified through the released csv files.

## Avatar Generation

We used the generators below, doing minor fixes to the code to improve quality and speed of generation:

- [GAGAvatar](https://github.com/xg-chu/GAGAvatar)
- [LivePortrait](https://github.com/KlingAIResearch/LivePortrait)
- [HunyuanPortrait](https://github.com/Tencent-Hunyuan/HunyuanPortrait)

We run the generators taking as inputs the driving videos and target images from the csv files described above.

> [!NOTE]
> For GAGAvatar and HunyuanPortrait, we used the **default configuration** the authors provide, only changing the paths to the corresponding inputs selected, as instructed. However, for LivePortrait, we set the value of the configuraton parameter `flag_relative_motion=False` to amplify expression driving strength.



### Computational Resources Needed for Dataset Generation

> [!IMPORTANT]
> The generators used to construct AVAPrintDB differ substantially in computational requirements. Check each of them separately for requirements (e.g. python version, Torch version, etc.).



These reported values correspond to the environment used to generate the released AVAPrintDB dataset and should be interpreted as representative operational indicators rather than universal benchmarks. Table below reports the time and peak GPU memory used to generate the avatars in a machine with the resources specified [here](../../README.md#computational-resources).


| Database | Generator | Avg. time per video (s) @ Resolution | Peak GPU memory usage
|---|---|:---:|:---:|
|CREMA-D|[GAGAvatar](https://github.com/xg-chu/GAGAvatar)|31 @ 480x360| ~7 GB
|CREMA-D|[LivePortrait](https://github.com/KlingAIResearch/LivePortrait)|11 @ 480x360| ~3 GB
|CREMA-D|[HunyuanPortrait](https://github.com/Tencent-Hunyuan/HunyuanPortrait)|66 @ 480x360| ~20 GB
|RAVDESS|[GAGAvatar](https://github.com/xg-chu/GAGAvatar)|35 @ 1280x720| ~7 GB
|RAVDESS|[LivePortrait](https://github.com/KlingAIResearch/LivePortrait)|14 @ 1280x720| ~3 GB
|RAVDESS|[HunyuanPortrait](https://github.com/Tencent-Hunyuan/HunyuanPortrait)|160 @ 1280x720| ~20 GB