from pathlib import Path
from typing import Tuple

import pandas as pd
from keras import layers, models, optimizers
from keras.src.legacy.preprocessing.image import (
    DataFrameIterator,
    ImageDataGenerator,
)
from sklearn.model_selection import train_test_split

import settings
from choices import TargetsChoice


class DataPreparation:
    def _get_train_image_path_target(self) -> pd.DataFrame:
        cat_images_path = Path(settings.IMAGE_PATH_TRAIN_CAT)
        dog_images_path = Path(settings.IMAGE_PATH_TRAIN_DOG)

        paths_and_targets = [
            (cat_images_path, TargetsChoice.CAT),
            (dog_images_path, TargetsChoice.DOG),
        ]

        full_paths = []
        targets = []

        for path, target in paths_and_targets:
            files = list(path.glob("*.jpg"))
            for file_path in files:
                full_paths.append(str(file_path))
                targets.append(target)

        dataset = pd.DataFrame(
            {
                "image_path": full_paths,
                "target": targets,
            }
        )

        return dataset

    def get_train_and_test(
        self,
    ) -> Tuple[DataFrameIterator, DataFrameIterator]:
        dataset = self._get_train_image_path_target()
        dataset_train, dataset_test = train_test_split(
            dataset, test_size=0.2, random_state=42
        )

        train_datagen = ImageDataGenerator(
            rotation_range=15,
            rescale=1.0 / 255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )
        train_data_generator = train_datagen.flow_from_dataframe(
            dataframe=dataset_train,
            x_col="image_path",
            y_col="target",
            target_size=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT),
            class_mode="binary",
            batch_size=32,
        )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_data_generator = test_datagen.flow_from_dataframe(
            dataframe=dataset_test,
            x_col="image_path",
            y_col="target",
            target_size=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT),
            class_mode="binary",
        )

        return (
            train_data_generator,
            test_data_generator,
        )

    def _get_vlidate_image_path_target(self) -> pd.DataFrame:
        cat_images_path = Path(settings.IMAGE_PATH_TRAIN_CAT)
        dog_images_path = Path(settings.IMAGE_PATH_TRAIN_DOG)

        paths_and_targets = [
            (cat_images_path, TargetsChoice.CAT),
            (cat_images_path, TargetsChoice.DOG),
        ]

        full_paths = []
        targets = []

        for path, target in paths_and_targets:
            if target == TargetsChoice.DOG:
                files = list(path.glob("*.jpg"))
            else:
                files = list(path.glob("*.jpg"))

            for file_path in files:
                full_paths.append(str(file_path))
                targets.append(target)

        dataset = pd.DataFrame(
            {
                "image_path": full_paths,
                "target": targets,
            }
        )

        return dataset

    def get_validate_data(self) -> DataFrameIterator:
        dataset = self._get_vlidate_image_path_target()

        validate_datagen = ImageDataGenerator(rescale=1.0 / 255)
        validate_data_generator = (
            validate_datagen.flow_from_dataframe(
                dataframe=dataset,
                x_col="image_path",
                y_col="target",
                target_size=(
                    settings.IMAGE_WIDTH,
                    settings.IMAGE_HEIGHT,
                ),
                class_mode="binary",
                shuffle=False,
            )
        )

        return validate_data_generator


def initialize_deep_learning_model() -> models.Sequential:
    model = models.Sequential()
    model.add(
        layers.Input(
            shape=(settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT, 3)
        )
    )

    model.add(layers.Conv2D(64, (2, 2), activation="elu"))
    model.add(layers.Conv2D(64, (3, 3), activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (2, 2), activation="elu"))
    model.add(layers.Conv2D(64, (3, 3), activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2, 2))

    model.add(layers.Conv2D(64, (2, 2), activation="elu"))
    model.add(layers.Conv2D(64, (3, 3), activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (2, 2), activation="elu"))
    model.add(layers.Conv2D(64, (3, 3), activation="elu"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="elu"))
    model.add(layers.Dense(256, activation="elu"))

    model.add(layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=0.0001),
        metrics=["accuracy"],
    )

    return model
