from keras.models import load_model
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

# === Load the trained model ===
model = load_model("cats_vs_dogs_model.keras")

# === Load training data ===
train_data = image_dataset_from_directory(
    "dataset/training_set",
    labels='inferred',
    label_mode='binary',
    image_size=(150, 150),
    batch_size=32,
    shuffle=True
)

# === Load validation data ===
val_data = image_dataset_from_directory(
    "dataset/test_set",
    labels='inferred',
    label_mode='binary',
    image_size=(150, 150),
    batch_size=32,
    shuffle=True
)

train_loss, train_accuracy = model.evaluate(train_data)
val_loss, val_accuracy = model.evaluate(val_data)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

labels = ['Train', 'Validation']
accuracies = [train_accuracy * 100, val_accuracy * 100]
losses = [train_loss, val_loss]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

ax[0].bar(labels, accuracies)
ax[0].set_title("Accuracy (%)")
ax[0].set_ylim(0, 100)
ax[0].set_ylabel("Accuracy")

ax[1].bar(labels, losses, color='orange')
ax[1].set_title("Loss")
ax[1].set_ylabel("Loss")

plt.suptitle("Model Evaluation")
plt.tight_layout()
plt.show()
