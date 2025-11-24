import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import json
from pathlib import Path

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


class StanceClassifierTrainer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.model = None

    def prepare_data(self):
        """Prepare data generators with augmentation"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2,
        )

        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            subset="training",
            classes={"good_stance": 0, "bad_stance": 1},
        )

        self.val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="binary",
            subset="validation",
            classes={"good_stance": 0, "bad_stance": 1},
        )

    def build_model(self):
        """Build a custom CNN model"""
        # Using MobileNetV2 as base for efficiency
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet"
        )

        # Freeze base model layers initially
        base_model.trainable = False

        # Build model
        inputs = keras.Input(shape=(*IMG_SIZE, 3))

        # Data augmentation layers
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)

        # Preprocessing for MobileNetV2
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

        # Base model
        x = base_model(x, training=False)

        # Custom top layers
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        # Output layer (binary classification)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        self.model = keras.Model(inputs, outputs)

        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )

    def train(self):
        """Train the model with callbacks"""
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=5, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                "best_model.h5", monitor="val_accuracy", save_best_only=True, mode="max"
            ),
        ]

        # Train model
        history = self.model.fit(
            self.train_generator,
            epochs=EPOCHS,
            validation_data=self.val_generator,
            callbacks=callbacks,
        )

        return history

    def fine_tune(self):
        """Fine-tune the base model"""
        # Unfreeze base model
        base_model = self.model.layers[4]
        base_model.trainable = True

        # Freeze early layers
        for layer in base_model.layers[:100]:
            layer.trainable = False

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.Precision(), keras.metrics.Recall()],
        )

        # Continue training
        history_fine = self.model.fit(
            self.train_generator,
            epochs=20,
            validation_data=self.val_generator,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
        )

        return history_fine

    def export_model(self, export_path="models/stance_classifier"):
        """Export model for web deployment"""
        # Save Keras model
        self.model.save(f"{export_path}.h5")

        # Convert to TensorFlow.js format
        import tensorflowjs as tfjs

        tfjs.converters.save_keras_model(self.model, export_path + "_tfjs")

        # Save metadata
        metadata = {
            "classes": ["good_stance", "bad_stance"],
            "image_size": IMG_SIZE,
            "model_type": "mobilenetv2_custom",
            "accuracy": float(self.model.evaluate(self.val_generator)[1]),
        }

        with open(f"{export_path}_metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"Model exported to {export_path}")


def main():
    # Initialize trainer
    trainer = StanceClassifierTrainer("data")

    # Prepare data
    print("Preparing data...")
    trainer.prepare_data()

    # Build model
    print("Building model...")
    trainer.build_model()

    # Train model
    print("Training model...")
    history = trainer.train()

    # Fine-tune
    print("Fine-tuning model...")
    history_fine = trainer.fine_tune()

    # Export model
    print("Exporting model...")
    trainer.export_model()

    # Print final metrics
    final_eval = trainer.model.evaluate(trainer.val_generator)
    print(f"\nFinal Validation Metrics:")
    print(f"Loss: {final_eval[0]:.4f}")
    print(f"Accuracy: {final_eval[1]:.4f}")
    print(f"Precision: {final_eval[2]:.4f}")
    print(f"Recall: {final_eval[3]:.4f}")


if __name__ == "__main__":
    main()
