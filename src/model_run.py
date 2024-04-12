import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from keras.models import load_model

from choices import TargetsChoice
from services import DataPreparation

model = load_model("model.h5")
validate_data_generator = DataPreparation().get_validate_data()
predictions = model.predict(validate_data_generator)

correct_predictions = 0
incorrect_predictions = 0
image_paths = validate_data_generator.filenames

incorrect_images = []

for path, prediction in zip(image_paths, predictions):
    actual_label = (
        TargetsChoice.DOG if "dog" in path else TargetsChoice.CAT
    )
    predicted_label = (
        TargetsChoice.DOG if prediction >= 0.5 else TargetsChoice.CAT
    )

    if actual_label == predicted_label:
        correct_predictions += 1
    else:
        incorrect_predictions += 1
        incorrect_images.append(path)

    print(f"{path}: {predicted_label} | {prediction}")
    print("________________________")

print(f"Количество правильных предсказаний: {correct_predictions}")
print(
    f"Количество неправильных предсказаний: {incorrect_predictions}"
)
if incorrect_images:
    print("Изображения с неправильными предсказаниями:")
    for img in incorrect_images:
        print(img)
