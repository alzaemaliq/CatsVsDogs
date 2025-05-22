import tensorflow
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.layers import Rescaling
import matplotlib.pyplot as plt

train_data = image_dataset_from_directory(
    "dataset/training_set",
    labels = 'inferred',
    label_mode = 'binary',
    image_size = (150, 150),
    batch_size = 32,
    shuffle = True
)

val_data = image_dataset_from_directory(
    "dataset/test_set",
    labels = 'inferred',
    label_mode = 'binary',
    image_size = (150, 150),
    batch_size = 32,
    shuffle = True
)

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1)
])


AUTOTUNE = tensorflow.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

model = Sequential([
    data_augmentation,
    Rescaling(1./255, input_shape=(150, 150, 3)),

    Conv2D(32, (3, 3), activation='relu'),  # 3 = RGB channels
    MaxPooling2D(2, 2),  # Shrinks image size, keeps important info

    Conv2D(64, (3, 3), activation='relu'),  # Deeper filters, finds more patterns
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),  # Deeper filters, finds more patterns
    MaxPooling2D(2, 2),

    Flatten(),  # Turns 2D image data into 1D for the final layers

    Dense(64, activation='relu'),  # Fully connected layer to learn features
    Dropout(0.3), #randomly disables 30% of the neurons during runtime
    Dense(1, activation='sigmoid')  # Output: 0 or 1 (cat or dog)
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

early_stop = EarlyStopping(

    monitor = 'val_accuracy',
    patience = 5,
    restore_best_weights = True
)

history = model.fit(
    train_data,
    validation_data = val_data,
    epochs = 30,
    callbacks = [early_stop]
)

plt.plot(history.history['accuracy'], label='Train Acc')       
plt.plot(history.history['val_accuracy'], label='Val Acc')    
plt.legend()
plt.title('Accuracy over Epochs')                           
plt.show()

model.save('cats_vs_dogs_model.keras')