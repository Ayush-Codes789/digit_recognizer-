import pandas as pd
import numpy as np
import tensorflow 
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv(r"C:\Users\ASUS\OneDrive\Attachments\Documents\Desktop\python\mnist_train_small.csv")
y = df.iloc[:, 0].values
x = df.iloc[:, 1:].values

# Normalize
x = x / 255.0

# Reshape for CNN: (samples, 28,28,1)
x = x.reshape(-1, 28, 28, 1)

# Train-test split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Build CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=64)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_val, np.argmax(model.predict(x_val)
# Save model
model.save("mnist_cnn.h5")
