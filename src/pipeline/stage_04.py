import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from image_classifier import logger

STAGE_NAME = "Prediction"
class PredictPipeline:
    def __init__(self):
        self.folder_path = os.path.join("src", "image_classifier", "components", "predict")
        self.class_indices_path = os.path.join("artifacts", "train", "class_indices.json")

    def main(self):
        model = load_model(os.path.join("artifacts", "train", "model.h5"))
        image_files = [f for f in os.listdir(self.folder_path) if os.path.isfile(os.path.join(self.folder_path, f))]
        selected_images = random.sample(image_files, 3)

        for image_name in selected_images:
            image_path = os.path.join(self.folder_path, image_name)
            test_image = image.load_img(image_path, target_size=(224, 224))
            test_image_array = image.img_to_array(test_image)
            test_image_array = np.expand_dims(test_image_array, axis=0)
            results = np.argmax(model.predict(test_image_array), axis=1)
            print(f"Prediction for {image_name}: {results}")

            if results[0] == 1:
                prediction = "Cat"
            else:
                prediction = "Not a Cat"

            print(f"Image: {image_name}, Prediction: {prediction}")

            plt_image = image.array_to_img(test_image_array[0])
            plt.imshow(plt_image)
            plt.title(f"Prediction: {prediction}")
            plt.axis('off')
            plt.show()

if __name__ == '__main__':
    try:
        logger.info(f"starting...{STAGE_NAME}")
        obj = PredictPipeline()
        obj.main()
        logger.info(f"completed...{STAGE_NAME}")
    except Exception as e:
        raise e
