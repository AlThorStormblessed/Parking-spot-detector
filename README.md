# Parking-spot-detector

This project is a real-time parking space detection system that uses computer vision and a custom-trained convolutional neural network (CNN) to identify free and occupied parking slots from CCTV footage or video files. It is ideal for smart parking systems and can be adapted to various layouts through manual configuration.

---

## Features

- Real-time video analysis using CCTV feeds or pre-recorded video
- Custom CNN classifier trained from scratch using parking data
- Visual marking of parking slot status (green for free, red for occupied)
- Dynamic ticket pricing model based on slot availability
- Manual setup of parking slot coordinates via mouse click
- Backup logic using pixel-based grayscale detection for robustness
- Easily extendable to new parking layouts with minimal code change

---

## Demo



https://github.com/user-attachments/assets/c94c33d5-19b9-4696-b070-f830a12adf17



---

## How It Works

1. **Slot Setup**
   - On running the script, click any two points to define the rectangle size of parking slots.
   - Then click on the top-left corners of each individual slot.
   - Press `'a'` to begin the detection loop.

2. **Detection Logic**
   - Each slot is cropped and resized to 128x128.
   - A custom CNN model classifies the image as free or occupied.
   - If the classification is uncertain or fails, pixel-count-based detection acts as a fallback.

3. **Pricing Model**
   - Ticket price is computed dynamically using a sigmoid-based pricing formula that scales with how full the lot is.

4. **Controls**
   - `'a'`: Start detection
   - `'z'`: Remove the last marked slot
   - `'q'`: Quit

---

## Model Details

The classification model is a simple CNN architecture built using Keras:

```python
model = Sequential([
    BatchNormalization(),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(128, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')
])
```

- Trained on custom-labeled images of "Free" and "Parked" slots.

---

