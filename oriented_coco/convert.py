import json
import os

# Specify the path to the annotation files
train_annotations_file = "annotations/instances_train2017.json"
val_annotations_file = "annotations/instances_val2017.json"

# Specify the output directories for train and val JSON files
train_output_dir = "train_bboxes"
val_output_dir = "val_bboxes"

# Create the output directories if they don't exist
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(val_output_dir, exist_ok=True)

# Function to process annotations and save JSON files
def process_annotations(annotations_file, output_dir):
    with open(annotations_file) as f:
        annotations = json.load(f)

    for image in annotations["images"]:
        image_id = image["id"]
        image_file_name = image["file_name"]
        image_height = image["height"]
        image_width = image["width"]

        image_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_file_name,
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }

        for annotation in annotations["annotations"]:
            if annotation["image_id"] == image_id:
                category_id = annotation["category_id"]
                category_name = next(cat["name"] for cat in annotations["categories"] if cat["id"] == category_id)
                bbox = annotation["bbox"]

                shape = {
                    "label": category_name,
                    "points": [
                        [bbox[0], bbox[1]],
                        [bbox[0] + bbox[2], bbox[1]],
                        [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                        [bbox[0], bbox[1] + bbox[3]]
                    ],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }

                image_json["shapes"].append(shape)

        output_file = os.path.join(output_dir, f"{image_id}.json")
        with open(output_file, "w") as f:
            json.dump(image_json, f, indent=2)

# Process train annotations
process_annotations(train_annotations_file, train_output_dir)

# Process val annotations
process_annotations(val_annotations_file, val_output_dir)
