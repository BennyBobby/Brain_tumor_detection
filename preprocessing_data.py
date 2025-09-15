import os
import cv2
import numpy as np

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
IMG_SIZE = (128, 128)


def process_data(raw_path: str, processed_path: str, size: tuple):
    """
    This function processes raw images from a source folder and saves them to a destination folder.
    It iterates through class folders in the source path, resizes
    and normalises each image, finally saves it as a .npy file
    in the destination path.

    Args:
        raw_path (str): The path to the source folder containing raw images
        processed_path (str): The path to the folder where the processed images
                              will be saved.
        size (tuple): A pair (width, height) representing the final image size.
    """
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    for tumour_type in os.listdir(raw_path):
        tumour_type_path = os.path.join(raw_path, tumour_type)
        processed_tumour_type_path = os.path.join(processed_path, tumour_type)
        if not os.path.exists(processed_tumour_type_path):
            os.makedirs(processed_tumour_type_path)
        for tumour_image in os.listdir(tumour_type_path):
            image_path = os.path.join(tumour_type_path, tumour_image)
            img = cv2.imread(image_path)
            processed_img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            processed_img = processed_img.astype("float") / 255.0
            np.save(
                os.path.join(
                    processed_tumour_type_path, tumour_image.split(".")[0] + ".npy"
                ),
                processed_img,
            )


if __name__ == "__main__":
    process_data(
        os.path.join(RAW_PATH, "Testing"),
        os.path.join(PROCESSED_PATH, "Testing"),
        IMG_SIZE,
    )
    print("Processing testing data done")
    process_data(
        os.path.join(RAW_PATH, "Training"),
        os.path.join(PROCESSED_PATH, "Training"),
        IMG_SIZE,
    )
    print("Processing training data done")
