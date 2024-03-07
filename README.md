# Exploring Training Pair-Generation Strategies for Deep Metric Learning for Floor Plan Retrieval

This repository includes some of the code from my thesis about floor plan retrieval, refactored to make it easier to use.

Thesis report: [Exploring Training Pair-Generation Strategies for Deep Metric Learning for Floor Plan Retrieval](http://resolver.tudelft.nl/uuid:68d3f999-c6f6-45ca-8209-d70a1fa00ef5)

Currently includes the code to download already trained models from wandb, and use them to retrieve similar floor plan images.

## Setting up environment

The python packages that are needed to run can be found in `environment.yml`. You can use conda/mamba to set it up, e.g. by running:

```
micromamba create -f environment.yml
```

## Retrieval using trained model

### Steps:
1. Prepare images
    1. Download RPLAN dataset
    2. Perform preprocessing
3. Download model
4. Predict embeddings
5. Use embeddings for retrieval

The [`retrieval-examples.ipynb`](retrieval-examples.ipynb) notebook also shows the relevant commands to run.

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


### Predict embeddings

Use the script `predict_embeddings.py` to save embeddings for each image.

#### Choose model to use

The `predict_embeddings.py` downloads one of the model checkpoints from wandb. The following checkpoints are available:

|Artifact name | Training pairs method | Dataset type |
---------------|----------------------|---------------- |
|model-is47i879:best | GED <=2 pairs | CONSISTENT_WALL_THICKNESS |
|model-ud4h7xj6:best | GED <=2 pairs | CATEGORY_CHANNEL |
|model-uc18eq89:best | GED <=2 pairs | RPLANPY |
|model-kbjnxwp1:best | SSIG pairs    | CONSISTENT_WALL_THICKNESS |
|model-mvau6ufl:best | IoU pairs     | CONSISTENT_WALL_THICKNESS |
|model-ujlo4ip0:best | GeomPerturb + masking   | CONSISTENT_WALL_THICKNESS |
|model-pkzm3f78:best | GeomPerturb | CONSISTENT_WALL_THICKNESS |
|model-rkm0f3ce:best | GeomPerturb w/o img augmentations | CONSISTENT_WALL_THICKNESS |
|model-8yzoiwmu:best | GED == 0 pairs | CONSISTENT_WALL_THICKNESS |
|model-8yzoiwmu:best | GED == 0 pairs + masking | CONSISTENT_WALL_THICKNESS |
|model-fkj6hn29:best | GED <=3 pairs | CONSISTENT_WALL_THICKNESS |

The report showed that GED <=2 pairs generally performed best. The CONSISTENT_WALL_THICKNESS dataset type uses a preprocessing approach that draws the floorplans with a consistent wall thickness. This preprocessing however only works on ~2/3 of RPLAN floor plans. The model trained on CATEGORY_CHANNEL uses the category channel of the rplan images directly, and RPLANPY uses the rplanpy library for drawing floor plan images.

### Retrieval

The `retrieval-examples.ipynb` shows how to retrieve similar floor plans from the dataset. By either:
- retrieving by rplan id
- prodiving an image to use as query

Additionally, the notebook shows what command to run for preprocessing the data and predicting features.