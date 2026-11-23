# Research Project: Generalization of Skin Lesion Classifiers

University research project on machine learning for disease detection from medical images.

## Research question

How much does the performance of a deep learning classifier trained on dermoscopic skin lesion images (HAM10000) drop when it is tested on clinical smartphone images (PAD-UFES-20), and can data augmentation and class balancing reduce this drop?

## Data (not included in the repository)

- **HAM10000** (dermoscopic images, training and internal test): Harvard Dataverse, doi 10.7910/DVN/DBW86T. Put it in `data/ham10000/`.
- **PAD-UFES-20** (smartphone images, external test): https://data.mendeley.com/datasets/zr7vgbcyr2. Put it in `data/pad_ufes_20/`.

The task is simplified to benign vs. malignant, using the four classes both datasets share: melanoma and basal cell carcinoma = malignant, nevus and benign keratosis (seborrheic keratosis in PAD-UFES-20) = benign. HAM10000 is split 70/15/15 by lesion, so no lesion appears in two sets. PAD-UFES-20 is only used for testing.

## Method

A ResNet-50 pretrained on ImageNet is fine-tuned in four configurations: A (baseline), B (data augmentation), C (class weights), D (both). Each is trained with several seeds. The decision threshold is chosen on the HAM10000 validation set and applied unchanged to the test sets. The generalization gap is the internal score minus the external score (AUC, balanced accuracy, sensitivity, specificity).

## Structure

- `src/data.py`: loading the datasets, image preparation and augmentation
- `src/train.py`: model, training and evaluation
- `src/experiment.py`: running one configuration, threshold choice, external test
- `notebooks/01_data_exploration.ipynb`: data loading, labels and split
- `notebooks/02_proof_of_concept.ipynb`: first small training run
- `notebooks/03_run_experiments.ipynb`: configurations A to D
- `notebooks/04_external_test.ipynb`: test on PAD-UFES-20 and generalization gap
- `results/`: split file and result tables (CSV)

## How to run

1. Install the packages: `pip install -r requirements.txt`
2. Download the datasets and put them in `data/` as described above.
3. Run the notebooks in order (01, then 03, then 04).
4. In notebook 03, `QUICK = True` runs a small test on a CPU; `QUICK = False` runs the full experiment (5 seeds, all images), which needs a GPU (e.g. Google Colab).

## Status

Work in progress. The pipeline works end to end on a small quick run; the full runs and final results will be added here.