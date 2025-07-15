import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Paths
DATASET_PATH = "C:/Users/pooji/Desktop/Smart Sorting/dataset/"

MODEL_PATH = "model/fruit_classifier.h5"

# Data generator
datagen = ImageDataGenerator(
    validation_split=0.2,
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(DATASET_PATH, target_size=(224, 224),
                                        batch_size=32, subset='training', class_mode='binary')

val_gen = datagen.flow_from_directory(DATASET_PATH, target_size=(224, 224),
                                      batch_size=32, subset='validation', class_mode='binary')

# Transfer learning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=preds)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# Compile and train
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=5)

# Save model
os.makedirs("model", exist_ok=True)
model.save(MODEL_PATH)
