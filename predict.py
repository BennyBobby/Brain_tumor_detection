import torch
import cv2
import numpy as np
import os
from model import BrainTumorClassifier
import torch.nn.functional as F
import matplotlib.pyplot as plt

MODEL_PATH = "model/brain_tumor_classifier.pth"
PROCESSED_DATA_PATH = "data/processed/Training"
image_test_path = "data/raw/Testing/meningioma_tumor/image(6).jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = sorted(os.listdir(PROCESSED_DATA_PATH))
CLASS_MAP = {i: name for i, name in enumerate(class_names)}
NUM_CLASSES = len(class_names)
IMG_SIZE = (128, 128)


def preprocess_image(image_path: str, size: tuple = IMG_SIZE):
    """
    This function pre-processes an image for model inference.
    The image is resized, converted from BGR to RGB, normalised, and
    reformatted to the PyTorch tensor shape (1, C, H, W).

    Args:
        image_path (str): The image's filepath to be processed.
        size (tuple): The target size (Width, Height).

    Returns:
        torch.Tensor: The image as a float tensor.

    Raises:
        FileNotFoundError: If the image file cannot be read.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: file not found: {image_path}")
    else:
        processed_img = cv2.resize(img, dsize=size, interpolation=cv2.INTER_AREA)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        processed_img = processed_img.astype("float") / 255.0
        processed_img = np.transpose(processed_img, (2, 0, 1))
        processed_img = np.expand_dims(processed_img, axis=0)
        return torch.from_numpy(processed_img).float()


def predict_image(image_path: str, model: BrainTumorClassifier, class_map, device):
    """
    This function loads, pre-processes, and runs the image through the trained model to get a prediction.
    Gradient tracking is temporarily disabled for efficiency.

    Args:
        image_path (str): The image's filepath for inference.
        model (BrainTumorClassifier): The trained PyTorch model.
        class_map (dict): A dictionary mapping integer indices to class names.
        device (torch.device): The device (CPU or CUDA) where the computation is run.

    Returns:
        tuple: A tuple containing (predicted_class_name, confidence_score).
    """
    image_tensor = preprocess_image(image_path).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_index].item()
    predicted_class = class_map[predicted_class_index]

    return predicted_class, confidence


if __name__ == "__main__":
    model = BrainTumorClassifier(NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    if os.path.exists(image_test_path):
        try:
            predicted_class, confidence = predict_image(
                image_test_path, model, CLASS_MAP, device
            )
            print(
                f"The model predicted that the image is {predicted_class } at {confidence:.1%}"
            )
            img = cv2.imread(image_test_path)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(img)
            ax.set_title(
                f"Prediction: {predicted_class}, Confidence Score: {confidence:.1%}"
            )
            ax.axis("off")
            plt.show()
        except Exception as e:
            print(f"Error {e} during prediction")
    else:
        print(f"The model path is not found: {image_test_path}")
