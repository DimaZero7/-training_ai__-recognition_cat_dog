import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from matplotlib import pyplot as plt

from services import DataPreparation, initialize_deep_learning_model

model = initialize_deep_learning_model()
train_data, test_data = DataPreparation().get_train_and_test()

model_history = model.fit(
    train_data,
    epochs=50,
    validation_data=test_data,
)

plt.plot(model_history.history["loss"])
plt.plot(model_history.history["val_loss"])
plt.savefig("training_testing_correlation.png")

model.save("model.h5")
