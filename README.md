# Thesis code

This repository includes some of the code from my msc thesis, restructured to make it easier to use.

## Retrieval using trained model

### Steps:
- Prepare images
    - Download RPLAN dataset
    - Perform preprocessing
- Download model
- Predict embeddings
- Use embeddings for retrieval

### Download RPLAN dataset:
Download RPLAN from: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html#sec_Downloads. You have to fill in a Google form to request access from the authors who published the RPLAN dataset.

### Preprocessing

Models were trained on three different variants of RPLAN images:
- Preprocessed geometries for drawing with consistent wall thickness.
- Using category channel of RPLAN directly.
- Using [rplanpy](https://github.com/unaisaralegui/rplanpy) for drawing the floor plans.

For each method, you can use the script `rplan_preprocessing.py` by specifying the option you want to use. The output of the preprocessing script is a hugging face dataset containing an image for each floor plan.

Example usage:
```
python run_preprocessing.py method=CONSISTENT_WALL_THICKNESS rplan_dataset_path="path/to/rplan/floorplan_dataset"
```


### Download model

Download one of the model checkpoints from wandb. The following checkpoints are available:
- 
- 
- 


### Predict embeddings

Use the script `predict_embeddings.py` to save embeddings for each image.