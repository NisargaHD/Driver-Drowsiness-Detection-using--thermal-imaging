import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    return image

# Function to build the model
def build_model():
    # CNN for image data
    image_input = Input(shape=(64, 64, 1))
    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    
    # Dense layers for temperature data
    temp_input = Input(shape=(1,))
    y = Dense(32, activation='relu')(temp_input)
    y = Dense(16, activation='relu')(y)
    
    # Combine both
    combined = concatenate([x, y])
    z = Dense(256, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(128, activation='relu')(z)
    z = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[image_input, temp_input], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to load and preprocess the dataset
def load_dataset():
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

    train_generator = datagen.flow_from_directory(
        'dataset_path',  # Replace with actual dataset path
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        'dataset_path',  # Replace with actual dataset path
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

# Function to train the model
def train_model():
    train_generator, validation_generator = load_dataset()
    model = build_model()
    history = model.fit(
        train_generator,
        epochs=30,  # Increase the number of epochs for better training
        validation_data=validation_generator
    )
    model.save('drowsiness_detection_model.h5')
    return model, history

# Function to detect eyes in the image
def detect_eyes(image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)
    return eyes

# Placeholder function for yawning detection
def detect_yawning(landmarks):
    pass

# Function to predict drowsiness based on eye diameter and temperature
def predict_drowsiness(model, eyes, temperature, threshold=20):
    if len(eyes) == 0:
        return 'PERSON IS DROWSY', temperature
    else:
        eye_diameters = [np.mean((eye[2], eye[3])) for eye in eyes]
        avg_eye_diameter = np.mean(eye_diameters)
        label = 'PERSON IS DROWSY' if avg_eye_diameter < threshold else 'PERSON IS NOT DROWSY'
        if temperature > 37:  # Example temperature threshold
            label = 'PERSON IS DROWSY'
        return label, temperature

# Function to apply thermal color map to grayscale image
def apply_thermal_color_map(image):
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    thermal_image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    return thermal_image

# Function to estimate head temperature from thermal image
def estimate_head_temperature(thermal_image):
    return np.mean(thermal_image) / 255 * 50  # Example conversion

# Main function to run real-time prediction
def main():
    try:
        model = tf.keras.models.load_model('drowsiness_detection_model.h5')
    except:
        model, history = train_model()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    cap = cv2.VideoCapture(0)
    total_frames = 0
    correct_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eyes = detect_eyes(gray_frame)

        landmarks = None
        if detect_yawning(landmarks):
            label = 'PERSON IS DROWSY'
            temperature = 'N/A'
        else:
            thermal_frame = apply_thermal_color_map(gray_frame)
            head_temperature = estimate_head_temperature(thermal_frame)
            label, temperature = predict_drowsiness(model, eyes, head_temperature)

        display_text = f"{label}, Temp: {temperature:.2f} °C"  # Create display text

        color = (0, 155, 255) if label == 'PERSON IS NOT DROWSY' else (0, 0, 0)
        cv2.putText(thermal_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5 if label == 'PERSON IS NOT DROWSY' else 1, color, 3)

        cv2.imshow("Driver Drowsiness Detection", thermal_frame)

        # Update accuracy
        total_frames += 1
        if label == 'Person is Not Drowsy':  # Assuming 'Not Drowsy' is correct classification
            correct_frames += 1

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    overall_accuracy = (correct_frames / total_frames) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

if __name__ == "__main__":
    main()