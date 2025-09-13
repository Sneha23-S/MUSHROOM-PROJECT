import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint

# =========================
# 1. Dataset Paths
# =========================
base_dir = os.getcwd()   # current working directory (Mushroom)
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# =========================
# 2. Data Generators
# =========================
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# =========================
# 3. Model Definition (Xception)
# =========================
base_model = Xception(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional base
for layer in base_model.layers:
    layer.trainable = False

x = Flatten()(base_model.output)
output_layer = Dense(train_generator.num_classes, activation="softmax")(x)  # classes = num folders in train/

model = Model(inputs=base_model.input, outputs=output_layer)

# =========================
# 4. Compile Model
# =========================
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# =========================
# 5. Train Model
# =========================
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

checkpoint = ModelCheckpoint("best_model.h5", monitor="val_accuracy", save_best_only=True, verbose=1)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[checkpoint]
)

# =========================
# 6. Model Summary
# =========================
model.summary()

# =========================
# 7. Load Best Model and Evaluate
# =========================
best_model = load_model("best_model.h5")
loss, acc = best_model.evaluate(validation_generator, verbose=1)
print(f"📊 Validation Accuracy: {acc*100:.2f}%")
print(f"📉 Validation Loss: {loss:.4f}")

# =========================
# 8. Predict on Single Image
# =========================
class_labels = list(train_generator.class_indices.keys())
print("Class labels:", class_labels)

# Example test image path (pick one image from test set)
img_path = os.path.join(test_dir, class_labels[0], os.listdir(os.path.join(test_dir, class_labels[0]))[0])

img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

predictions = best_model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions)

print(f"✅ Predicted Class: {class_labels[predicted_class]} (Confidence: {confidence*100:.2f}%)")

# =========================
# 9. Plot Training History
# =========================
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()
