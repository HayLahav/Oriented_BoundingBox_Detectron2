import os
import json
from PIL import Image

def unite_annotations_and_images(annotations_folder, images_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over the annotation files
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith(".json"):
            annotation_path = os.path.join(annotations_folder, annotation_file)
            image_name = annotation_file.replace(".json", "")

            # Find the corresponding image file
            image_extensions = [".jpg", ".jpeg", ".png"]
            image_path = None
            for ext in image_extensions:
                temp_path = os.path.join(images_folder, image_name + ext)
                if os.path.isfile(temp_path):
                    image_path = temp_path
                    break

            if image_path:
                # Read the annotation file
                with open(annotation_path, "r") as f:
                    annotation_data = json.load(f)

                # Update the annotation file with image information
                annotation_data["imagePath"] = os.path.basename(image_path)
                annotation_data["imageData"] = None

                # Get the image dimensions
                with Image.open(image_path) as img:
                    width, height = img.size
                annotation_data["imageHeight"] = height
                annotation_data["imageWidth"] = width

                # Save the updated annotation file in the output folder
                output_annotation_path = os.path.join(output_folder, annotation_file)
                with open(output_annotation_path, "w") as f:
                    json.dump(annotation_data, f, indent=2)

                # Copy the image file to the output folder
                output_image_path = os.path.join(output_folder, os.path.basename(image_path))
                os.replace(image_path, output_image_path)

# Example usage
annotations_folder =  "/home/tauproj5/project_venv/DiffusionDet/blog_detectron2/blog/coco/train_bboxes"
images_folder = "/home/tauproj5/project_venv/DiffusionDet/blog_detectron2/blog/coco/train2017"
output_folder = "/home/tauproj5/project_venv/DiffusionDet/blog_detectron2/blog/coco/combined"

unite_annotations_and_images(annotations_folder, images_folder, output_folder)
