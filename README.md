<h1 align="center">
  <a href="https://arxiv.org/pdf/2603.26934">
    Leveraging Avatar Fingerprinting: A Multi-Generator Photorealistic Talking-Head Public Database and Benchmark
  </a>
</h1>



<h3 align="center">
    <a href='https://arxiv.org/pdf/2603.26934'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
    <a href='#request-for-access'><img src='https://img.shields.io/badge/Data and Code-Request Access-blue'></a> &nbsp; 
</h3>

<h5 align="center">
    <a href="https://scholar.google.com/citations?user=xYLElMkAAAAJ&hl=es">Laura Pedrouzo Rodriguez</a> &emsp;
    <a href="https://scholar.google.com/citations?user=Nq3NyHYAAAAJ&hl=en">Luis F. Gomez</a> &emsp;
    <a href="https://rubentolosana.github.io/">Ruben Tolosana Moranchel</a>  &emsp;<br>
    <a href="https://scholar.google.com/citations?user=KYMQ0tsAAAAJ&hl=es">Ruben Vera Rodriguez</a>  &emsp;
    <a href="https://scholar.google.com/citations?hl=es&user=u0e5cXkAAAAJ">Roberto Daza</a>  &emsp;
    <a href="https://scholar.google.com/citations?user=yRP16B4AAAAJ&hl=es">Aythami Morales</a>  &emsp;
  <a href="https://scholar.google.com/citations?user=HbG_NOoAAAAJ&hl=en">Julian Fierrez</a>  &emsp;
    <br><br>
    Universidad Autónoma de Madrid, UAM
</h5>

<h3 align="center">
<img height="45" alt="image" src="assets/uamlogo.png" />    &emsp;  <img  height="35" alt="image" src="assets/BiometricsAI_logo.png" /> 
</h3>

Code will be released soon!

# About

This is the official repository for the paper [Leveraging Avatar Fingerprinting: A Multi-Generator Photorealistic Talking-Head Public Database and Benchmark](https://arxiv.org/pdf/2603.26934). 

We are releasing a **new talking-head avatar video database and benchmark**.

# Avatar Database: AVAPrintDB

We release a dataset of +65,000 videos of photorealistic avatars, generated using [GAGAvatar](https://github.com/xg-chu/GAGAvatar) (Neurips 2024), [LivePortrait](https://github.com/KlingAIResearch/LivePortrait) (2025), and [HunyuanPortrait](https://github.com/Tencent-Hunyuan/HunyuanPortrait) (CVPR 2025). We take real videos from [RAVDESS](https://zenodo.org/records/1188976) and [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) datasets as the base target identities and driving videos to generate our dataset. We include <span style="color: green">**genuine**</span> and <span style="color: red">**impostor**</span> videos. 
<table align="center">
  <tr>
    <td><img src="assets/examples_avatars.gif" width="100%"/></td>
    <td><img src="assets/generation.png" width="100%"/></td>
  </tr>
</table>

### Database Structure

The AVAPrintDB root folder contains these subfolders: `videos`, where you can find the mp4 files, `metadata` where you can find dataset information,  and `eval_files` where you can find the evaluation files used for the benchmark.

```
AVAPrintDB
├── eval_files
│   ├── evaluation_pairs_CREMA-D_GAGA.csv
│   ├── evaluation_pairs_CREMA-D_HUNY.csv
│   ├── evaluation_pairs_CREMA-D_LIVE.csv
│   ├── evaluation_pairs_RAVDESS_GAGA.csv
│   ├── evaluation_pairs_RAVDESS_HUNY.csv
│   └── evaluation_pairs_RAVDESS_LIVE.csv
├── metadata
│   ├── avaprintdb_metadata.csv
│   ├── cremad_metadata.csv
│   └── ravdess_metadata.csv
└── videos
    ├── DEV
    │   ├── GAGA
    │   ├── HUNY
    │   └── LIVE
    └── TEST
        ├── GAGA
        ├── HUNY
        └── LIVE
```

#### Videos

In `videos` folder, you can find two subfolders: `dev` and `test`. The `dev` folder contains all the videos used for development (training and validation), and the `test` folder contains all the videos used for evaluation. The data was split so that each identity can only be present as driver or target in one of the subsets, either in `dev` or in `test`, i.e. test data contains ONLY unseen target and driver identities.

The naming convention used for the video files is:

    <target_id>--<driver_id>--<video_uuid_name>--<generator>.mp4

Where <target_id> and <driver_id> correspond to the real identities from the source datasets, <generator> correspond to the avatar generator used to generate the avatar (`GAGA` for GAGAvatar, `LIVE` for LivePortrait and `HUNY` for HunyuanPortrait), and <video_uuid_name> corresponds to a UUID obtained for each unique source video. The mappings between UUIDs and source videos are reported in the metadata files.

> **Example**
> 
> Avatar video with filename `C1061--C1005--063b1674-e708-5f9c-8f09-2f75f27a31aa--GAGA.mp4`, would correspond to an avatar generated using identity `1061` from CREMA-D as the target (appearance), identity `1005` from CREMA-D as the driver, using GAGAvatar generator with the CREMA-D video corresponding to the UUID `063b1674-e708-5f9c-8f09-2f75f27a31aa`.


#### Metadata

In this folder you can find these files:
- `avaprintdb_metadata.csv`: this file contains a list of each of the avatar videos in the database, indicating the source data used to generate it, and the split it corresponds to (`DEV`, `TEST`).
- `cremad_metadata.csv`: this file contains a list of all the crema-d videos used as source, with the soft-biometric data correspondin to the identity in the video, and the recording details (statement spoken, resolution, etc.).
- `ravdess_metadata.csv`: this file contains a list of all the ravdess videos used as source, with the soft-biometric data correspondin to the identity in the video, and the recording details (statement spoken, resolution, etc.).


#### Evaluation files

In this folder you can find one csv file per avatar generator and source dataset, which contain the pairs of "enrolment video" and "test video" with the corresponding label (1 if both correspond to the same driving identity, 0 if they don't). These are the evaluation files used for the benchmark.



# Proposed verification systems based on Foundation Models

To be released soon!



# Requesting access

If you want to gain access to the dataset, code and the pre-trained models, follow these steps:

1. Download [this](assets/AVAPrintDB_Agreement.pdf) license agreement, complete it and sign it.
2. Send an email to **[atvs@uam.es](mailto:atvs@uam.es)** with the following information:

   - ***Subject***: `[DATABASE: AVAPrintDB]`
   
   - ***Body***:  Include your full name, email, phone number, organization, postal address and the purpose for which you will use the database
   
   - ***Attachments***: The signed and scanned license agreement (PDF)
   
3. Once we receive the license agreement and review your request, you will receive an email with a username, password and instructions to download the data.
4. For more information about the request process, or if you encounter any trouble, please contact atvs@uam.es



# Contact

For more information contact Rubén Tolosana, associate professor at UAM at **[ruben.tolosana@uam.es](mailto:ruben.tolosana@uam.es)** or Laura Pedrouzo, PhD student at **[laura.pedrouzo@uam.es](mailto:laura.pedrouzo@uam.es)**
