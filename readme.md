**Brain Tumour Classification with PyTorch**
A Convolutional Neural Network (CNN) project built with PyTorch to classify brain MRI scans into different tumour types.

**Dataset**
This project is designed to work with the Brain MRI Images for Brain Tumour Classification dataset. This dataset typically contains separate directories for training and testing images, categorised by tumour type.


**Getting Started**

***1. Installation***

Install all necessary packages using the provided requirements.txt file. I used Python 3.13.5
```bash
pip install -r requirements.txt
```
***2. Data Setup***

Download the Dataset: https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri

Place the raw image files (including the Testing and Training folders) inside the data/raw directory.

***3. Data Pre-processing***
Run the preprocessing_data.py script:
```bash
python preprocessing_data.py
```
This script will create the data/processed/Training and data/processed/Testing directories containing the .npy files required for the PyTorch DataLoader.

***4. Model Training***

Run the training script:
```bash
python train.py
```
***5. Inference***
Update the image_test_path variable in predict.py if you want to test with another image. Then, run the prediction script:

```bash
python predict.py
```

**Sources**
https://www.kaggle.com/code/guslovesmath/tumor-classification-99-7-tensorflow-2-16 

https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html 