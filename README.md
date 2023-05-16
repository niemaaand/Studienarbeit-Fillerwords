# studienarbeit
This repository contains code and some trained models, developed 
in a student research project, to detect filler words ("uh", "uhm") 
in speech. 

## requirements
Project is implemented with Python 3.10. To install dependencies, use 
`pip install -r requirements.txt`. 

If you want to run the preprocessing, you need to have 
`ffmpeg` (https://ffmpeg.org/download.html) installed. 

To run the code, adapt the paths in 
`PathsForRunningInColabSingleton.path_to_saved_models` and 
`PathsForRunningInColabSingleton.dataset_path` according to your local system. 


## use as software
To run the filler recognition as software on a .wav-file, use 
`filler_detection_software/main.py`. Just adapt the variable *wav_path* in 
`evaluation/options_for_continuous_speech.py` so that it points to the .wav-file 
you want to get the fillers in. 

As a result, you will get a visualization, so make sure you have a 
monitor available. (Visualization part does not run in 
google-colaboratory (colab)). 

In this visualization, you have some usual audio-player-functionalities - 
but this audio-player is very rudimentary, so if you want to switch 
playing-mode (e.g. from usual playing to play only filler) press 
"stop"-button before pressing "play filler". 

## personal adaptions / retraining
To build and train your personal model, adapt the variable of type 
OptionsInformation in main.py to your preferred settings and 
adapt the paths in paths_for_running_in_colab.py to your system. 

## Execute data preprocessing

Data is not available in this repo. For data, refer to 
`https://zenodo.org/record/7121457#.ZF0LTHZByUn` (PodcastFillers.z01 -> can be unpacked 
with 7zip) and the preprocessing methods implemented in this project. 

Preprocessing is arranged in `main.py` by executing method `preprocess`. 

### created files during preprocessing

The following files are created during preprocessing and stored in clip-wav-folder. 

#### goneWrongCreatingClip.json

This file is produced during cutting clips.
It contains all clips (with their podcast), with which something went wrong during 
creation of this clip. Further explanation is shown in the following table. 


| Category | Effect                                                         | Reason                                      | Is Clip created |
|----------|----------------------------------------------------------------|---------------------------------------------|-----------------|
| 1        | Clip has a duration of 0 seconds or a size of 0 bytes          | Cutting clip from outside the podcast       | No              |
| 2        | Clip has approximately the specified duration, but not exactly | Usage of float. -> Should not occur anymore | Yes             |

Clips that are in this file but not in faults.json are of category 2. 


#### faults.json

This file is produced after cutting all clips. It contains only clips, which are in 
goneWrongCreatingClips.json, exist and do not have exactly the specified duration. 

The clips in this file can be removed from the data with the method 
`remove_clips_according_to_json()`. 

#### mp3FromBrokenClips.json

The clips contained in faults.json are sorted by their podcast-episodes (mp3-files). 

