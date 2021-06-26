# vn-cough-detection

# Table of contents
- [Training](#training)
- [Tracking experiment](#tracking-experiment)
- [Manage Artifact](#manage-artifact)
- [Custom dataset](#custom-dataset)


Training
========

Tracking experiment
==================

Manage artifact
==============

Custom dataset
==============
I have create a custome dataset name `AICoughDataset`, which located in `utils.dataset`. In order to use it, the dataset must follow this format:
```
    |—train
    |   |—images
    |   |   |—id1.jpg
    |   |   |—id2.jpg
    |   |   |— ...
    |   |—metadata_train.csv
    |—test
    |   |—images
    |   |   |—id1.jpg
    |   |   |—id2.jpg
    |   |   |— ...
    |   |—metadata_test.csv
    |
```
Where `images` dir will store all `.jpg` images and `metadata_train.csv` must contain `uuid`, `assessment_result`, like shown:

| uuid                                 | ... | assessment_result | ... |
|--------------------------------------|-----|-------------------|-----|
| 3284bcf1-2446-4f3a-ac66-14c76b294177 |     | 0                 |     |
| 431334e1-5946-4576-bb51-8e342ccc22b4 |     | 0                 |     |
| 8fe351e5-2274-4858-87e4-b8a3528cd78c |     | 1                 |     |
| eef40e1d-ee26-469b-b475-c767ccfefe0b |     | 1                 |     |