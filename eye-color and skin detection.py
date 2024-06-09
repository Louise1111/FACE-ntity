import sys
import os
import numpy as np
import cv2
import argparse
from tkinter import filedialog
import tkinter as tk
from PIL import Image, ImageTk
from mtcnn.mtcnn import MTCNN

detector = MTCNN()

class_name = (
    "Blue", "Blue Gray", "Brown", "Brown Gray", "Brown Black",
    "Green", "Green Gray", "Hazel", "Amber", "Gray", "Red", "Black","other"
)

EyeColor = {
    class_name[0]: ((166, 21, 50), (240, 100, 85)),    # Blue
    class_name[1]: ((166, 2, 25), (300, 20, 75)),      # Blue Gray
    class_name[2]: ((2, 20, 20), (40, 100, 60)),       # Brown
    class_name[3]: ((20, 3, 30), (65, 60, 60)),        # Brown Gray
    class_name[4]: ((0, 10, 5), (40, 40, 25)),         # Brown Black
    class_name[5]: ((60, 21, 50), (165, 100, 85)),    # Green
    class_name[6]: ((60, 2, 25), (165, 20, 65)),      # Green Gray
    class_name[7]: ((10, 10, 10), (90, 60, 60)),      # Hazel
    class_name[8]: ((0, 40, 60), (40, 90, 120)),      # Amber
    class_name[9]: ((80, 5, 80), (140, 40, 140)),     # Gray
    class_name[10]: ((0, 70, 70), (20, 255, 255)),    # Red
    class_name[11]: ((0, 0, 0), (30, 30, 30)),        # Black
}


ComplexionColorRange  = {
    "Fair": ((0, 10, 130), (20, 150, 255)),
    "Medium": ((0, 10, 80), (20, 160, 180)),
    "Dark": ((0, 10, 40), (20, 160, 100)),
    "Olive": ((10, 20, 50), (40, 150, 150)),
    "Pale": ((0, 5, 120), (20, 110, 230)),
    "Tan": ((5, 20, 80), (25, 130, 180)),
    "Light Brown": ((10, 30, 60), (25, 160, 180)),
    "Cocoa": ((0, 10, 20), (30, 90, 120)),
    "Ebony": ((0, 0, 0), (30, 30, 30)),
    "Porcelain": ((0, 5, 150), (15, 120, 255)),
    "Caramel": ((10, 40, 90), (30, 150, 180)),
    "Mahogany": ((0, 0, 30), (20, 70, 100)),
  
}

# Functions for color checking and identification

def check_color(hsv, color):
    return (hsv[0] >= color[0][0]) and (hsv[0] <= color[1][0]) and \
           (hsv[1] >= color[0][1]) and (hsv[1] <= color[1][1]) and \
           (hsv[2] >= color[0][2]) and (hsv[2] <= color[1][2])

def find_class(hsv):
    for i in range(len(class_name) - 1):
        if check_color(hsv, EyeColor[class_name[i]]):
            return i
    return len(class_name) - 1


def check_complexion_color(hsv, color_range):
    return (hsv[0] >= color_range[0][0]) and (hsv[0] <= color_range[1][0]) and \
           (hsv[1] >= color_range[0][1]) and (hsv[1] <= color_range[1][1]) and \
           (hsv[2] >= color_range[0][2]) and (hsv[2] <= color_range[1][2])

def find_complexion_color(hsv):
    for complexion, color_range in ComplexionColorRange.items():
        if check_complexion_color(hsv, color_range):
            return complexion
    return 'Other'

# Function for eye and complexion detection

def eye_and_complexion_detection(image):
    imgHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[0:2]
    imgMask = np.zeros((image.shape[0], image.shape[1], 1))
    eye_class = np.zeros(len(class_name), np.float64)
    complexion_class = {complexion: 0 for complexion in ComplexionColorRange.keys()}
    complexion_class['Other'] = 0

    result = detector.detect_faces(image)
    if result == []:
        print('Warning: Cannot detect any face in the input image!')
        return

    bounding_box = result[0]['box']
    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']

    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    eye_radius = eye_distance / 15  # approximate

    cv2.circle(imgMask, left_eye, int(eye_radius), (255, 255, 255), -1)
    cv2.circle(imgMask, right_eye, int(eye_radius), (255, 255, 255), -1)

    cv2.rectangle(image, (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (255, 155, 255), 2)

    cv2.circle(image, left_eye, int(eye_radius), (0, 155, 255), 1)
    cv2.circle(image, right_eye, int(eye_radius), (0, 155, 255), 1)

    for y in range(h):
        for x in range(w):
            if imgMask[y, x] != 0:
                eye_class[find_class(imgHSV[y, x])] += 1
                complexion_tone = find_complexion_color(imgHSV[y, x])
                complexion_class[complexion_tone] += 1

    total_vote = eye_class.sum()
  
    print("\n\nDominant Eye Color: ", class_name[np.argmax(eye_class)])
    print("\n**Eyes Color Percentage **")
    for i, eye in enumerate(class_name):
        percentage = round(eye_class[i] / total_vote * 100, 2)
        print(eye, ": ", percentage, "%")

    print("\n**Complexion Color Analysis**")
    for complexion_tone, count in complexion_class.items():
        print(complexion_tone, ": ", count)

    total_complexion_count = sum(complexion_class.values())
    dominant_complexion_percentage = {complexion_tone: count / total_complexion_count * 100 for complexion_tone, count in complexion_class.items()}

    print("\n**Dominant Complexion Color Percentage **")
   
    for complexion_tone, percentage in dominant_complexion_percentage.items():
        print(complexion_tone, ": ", percentage, "%")

    return image, eye_class, complexion_class



def process_and_display_image(file_path):
    # Clear existing labels
    for widget in root.winfo_children():
        if isinstance(widget, tk.Label):
            widget.pack_forget()

    if file_path:
        image = cv2.imread(file_path)
        processed_image, eye_class, complexion_class = eye_and_complexion_detection(image)

        # Convert to RGB format
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Resize the image if needed
        max_height = 400  
        if processed_image.shape[0] > max_height:
            scale_percent = max_height / processed_image.shape[0]
            width = int(processed_image.shape[1] * scale_percent)
            height = max_height
            processed_image = cv2.resize(processed_image, (width, height))

        img = Image.fromarray(processed_image)
        img = ImageTk.PhotoImage(image=img)

        panel = tk.Label(root, image=img)
        panel.image = img
        panel.pack()

        # Display dominant eye color and percentages
        eye_color = class_name[np.argmax(eye_class)]
        eye_percentage_labels = []

        for i, eye in enumerate(class_name):
            percentage = round(eye_class[i] / eye_class.sum() * 100, 2)
            eye_percentage_text = f"{eye}: {percentage}%"
            eye_percentage_labels.append(eye_percentage_text)

        eye_color_label = tk.Label(root, text=f"Dominant Eye Color: {eye_color}", font=("Helvetica", 14))
        eye_color_label.place(relx=1.0, x=-10, y=10, anchor='ne')  # Top right corner

        for i, eye_percentage in enumerate(eye_percentage_labels):
            label = tk.Label(root, text=eye_percentage, font=("Helvetica", 10))
            label.place(relx=1.0, x=-10, y=40 + 20 * i, anchor='ne')  # Top right corner

        # Display dominant complexion color and percentages at top left corner
        dominant_complexion_color = max(complexion_class, key=complexion_class.get)
        dominant_complexion_percentage = complexion_class[dominant_complexion_color] / sum(complexion_class.values()) * 100

        complexion_color_label = tk.Label(root, text=f"Dominant Complexion Color: {dominant_complexion_color}", font=("Helvetica", 14))
        complexion_color_label.place(x=10, y=10, anchor='nw')  # Top left corner

        for i, (complexion_color, count) in enumerate(complexion_class.items()):
            percentage = count / sum(complexion_class.values()) * 100
            complexion_percentage_text = f"{complexion_color}: {percentage:.2f}%"
            label = tk.Label(root, text=complexion_percentage_text, font=("Helvetica", 10))
            label.place(x=10, y=40 + 20 * i, anchor='nw')  # Top left corner

    else:
        print("No file selected.")

def upload_image():
    file_path = filedialog.askopenfilename()
    process_and_display_image(file_path)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Face-ntity: An eye color and complexion Identification System")
    root.geometry("800x600")
    root.configure(bg="lightblue")

    upload_button = tk.Button(root, text="Upload Image", command=upload_image, 
                              bg="gray", fg="black", font=("Helvetica", 12), 
                              relief=tk.RAISED, highlightbackground="black")
    upload_button.pack()

    root.mainloop()