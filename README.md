# CBIR_Explainability_Skin_Cancer
A **Tensorflow** implementation of a **CBIR Explainable Model** for the diagnosis of skin lesions.
**PyTorch** implementation coming soon....

![](./imgs)

## Requirements
Tensorflow version >= 2.3

Python >= 3.6

Scikit-learn

This code uses functions form tf.slim.

## Usage
1) Download the dataset tf recorders from **https://tinyurl.com/yd65v34z** and add to the **data** folder or create your own tf records and add them to the **same** folder

2) Train a hierarchical model using a specific fold, network, attention properties, batch_size, and number of epochs: python  model_train.py --tfrecord_train "data\Fold_1_T3\Training\train_full_norm.tfrecords" --tfrecord_val "data\Fold_1_T3\Validation\val_full_norm.tfrecords" --net "V" --feature_maps 512 --train_batch_size 20 --ratio 2 --how_many_training_steps 150 

## Reference

```
@inproceedings{barata2021,
  title={Improving the Explainability of Skin Cancer Diagnosis Using CBIR},
  author={Barata, Catarina and Santiago, Carlos},
  booktitle={Accepted for Publication in },
  year={2021}
}


```
