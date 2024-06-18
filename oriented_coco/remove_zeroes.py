import os
import json

def remove_leading_zeroes_in_anno_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            if 'imagePath' in data:
                image_path = data['imagePath']
                modified_image_path = image_path.lstrip('0')
                data['imagePath'] = modified_image_path
                
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=2)
    
    print("Leading zeroes removed from 'imagePath' in all annotation files.")

# Provide the path to your annotations folder
annotations_folder = '/home/tauproj5/project_venv/DiffusionDet/blog_detectron2/blog/coco/train_bboxes'
remove_leading_zeroes_in_anno_folder(annotations_folder)



def remove_leading_zeroes_from_images(folder_path):
    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        # Check if the file has a valid image extension
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Split the filename and extension
            name, ext = os.path.splitext(filename)
            
            # Remove leading zeroes from the filename
            modified_name = name.lstrip('0')
            
            # Construct the new filename
            new_filename = modified_name + ext
            
            # Construct the full file paths
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
    
    print("Leading zeroes removed from image filenames.")

# Provide the path to your images folder
images_folder = '/home/tauproj5/project_venv/DiffusionDet/blog_detectron2/blog/coco/train2017'
remove_leading_zeroes_from_images(images_folder)
