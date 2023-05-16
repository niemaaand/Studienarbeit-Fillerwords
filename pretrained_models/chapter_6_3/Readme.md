# Explanation - Results

## Retraining
Basis of retraining are the models in 
`pretrained_models/chapter_6_2/tcresnet_variation/8_1_DS`. 

The correlation is as follows:  

folder (here): file name (in `pretrained_models/chapter_6_2/tcresnet_variation/8_1_DS`)  

1: `model_dsmix_tcresnet_pool_drop15_tensor(0.9079, device='cuda_0').pt`   
2: `best_dsmix_tcresnet_pool_drop57_tensor(0.9025, device='cuda_0').pt`  
3: `best_dsmix_tcresnet_pool_drop11_tensor(0.9052, device='cuda_0').pt`

## Manually checked labels

The file `MasterCsvWithManuallyCheckedAnnotations.csv` contains the clips, whose 
annotations were checked manually in this project. If the annotation is "filler"/
"non_filler", then the original label was wrong. In this project, it was only 
worked on these two labels, additional labels were projected on 
"filler"/"non_filler". 

# Reproduce - Retraining
If you want to retrain a model, use `retrain_on_corrected_labels` in `main.py`.
Current paths are set to number 3 (section above). 

Subset files are available in this repo at 
`pretrained_models/chapter_6_3/checked_metadata/subsets`. 

For data acquisition refer to readme in root folder in this repo. 



