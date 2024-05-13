import os
import cv2
import json
import numpy as np

def get_labelme_dataset_function(labelme_directory, class_labels):
    def dataset_function():
        return labelme_directory_to_detectron_dataset(labelme_directory, class_labels)
    return dataset_function

def labelme_directory_to_detectron_dataset(directory, class_labels):
    files = os.listdir(directory)
    dataset_dicts = []
    
    for filename in files:
        if filename.endswith('.json'):
            path = os.path.join(directory, filename)
            with open(path, 'rt') as f:
                data = json.load(f)
                
                image_path = os.path.join(directory, data['imagePath'])
                height = data['imageHeight']
                width = data['imageWidth']
                
                annotations = []
                for shape in data['shapes']:
                    label = shape['label']
                    if label not in class_labels:
                        continue
                    
                    points = np.array(shape['points'], dtype=np.float32)
                    ((cx, cy), (w, h), a) = cv2.minAreaRect(points)
                    
                    if w < h:
                        h_temp = h
                        h = w
                        w = h_temp
                        a += 90
                    
                    a = (360 - a) % 360  # ccw [0, 360]
                    
                    # Clamp to [0, 90] and [270, 360]
                    if (a > 90) and (a <= 180):
                        a -= 180
                    elif (a > 180) and (a < 270):
                        a -= 180
                    
                    # Clamp to [-180, 180]
                    if a > 180:
                        a -= 360
                    
                    annotations.append({
                        "bbox_mode": 4,  # Oriented bounding box (cx, cy, w, h, a)
                        "category_id": class_labels.index(label),
                        "bbox": (cx, cy, w, h, a)
                    })
                
                image_id = os.path.splitext(filename)[0]
                dataset_dicts.append({
                    "file_name": image_path,
                    "height": height,
                    "width": width,
                    "image_id": image_id,
                    "annotations": annotations
                })
    
    return dataset_dicts
