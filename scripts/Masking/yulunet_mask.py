import cv2
import numpy as np
import os
from pathlib import Path

def mask_faces_yunet(input_dir, output_dir, model_path, mask_color=(0, 0, 0)):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    count=0
    
    face_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        (2000, 2000),  # Input size - can be adjusted
        0.50,         # conf. threshold
        0.1,         # NMS threshold, overlapping boxes
        20         # Top faces to detect
    )
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Error: Could not load image: {image_path}")
                continue  
            
            height, width = image.shape[:2]
            face_detector.setInputSize((width, height))
            
            _, faces = face_detector.detect(image)
            
            if faces is not None:
                for face in faces:
                    x, y, w, h = map(int, face[:4]) #get the main dims., map to int for image indexing
                    confidence = face[-1]
                    
                    padding_x = int(0.05 * w)  
                    padding_y = int(0.05 * h)
                    
                    x1 = max(x - padding_x, 0)
                    y1 = max(y - padding_y, 0)
                    x2 = min(x + w + padding_x, width)
                    y2 = min(y + h + padding_y, height)                    
                    image[y1:y2, x1:x2] = mask_color
            
            output_path = os.path.join(output_dir, f"masked_{filename}")
            cv2.imwrite(output_path, image)
            print(f"Processed: {filename}")
            count+=1
            print(f"Total images processed: {count}")

    print(f"Total images processed: {count}")

def main():
    input_directory = ""


    output_directory = ""
    model_path = "./face_detection_yunet_2023mar_int8bq.onnx"

    mask_color = (0, 0, 0)  # 
    mask_faces_yunet(input_directory, output_directory, model_path, mask_color)

if __name__ == "__main__":
    main()