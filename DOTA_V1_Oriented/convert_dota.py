import json
import os

def convert_dota_to_json(dota_path, output_path):
    # Iterate over the DOTA annotation files
    for anno_file in os.listdir(dota_path):
        if anno_file.endswith(".txt"):
            image_id = anno_file[:-4]  # Remove the ".txt" extension to get the image ID
            anno_path = os.path.join(dota_path, anno_file)
            
            # Create a JSON object for the image
            json_data = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [],
                "imagePath": f"{image_id}.png",
                "imageData": None,
                "imageHeight": 1024,  # Set the DOTA v1.0 image height
                "imageWidth": 1024    # Set the DOTA v1.0 image width
            }
            
            # Parse the DOTA annotation file
            with open(anno_path, "r") as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 10:  # Check if the line has the expected format
                        category = data[8]
                        points = [[float(data[0]), float(data[1])], [float(data[2]), float(data[3])],
                                  [float(data[4]), float(data[5])], [float(data[6]), float(data[7])]]
                        difficulty = int(data[9])  # Extract the difficulty level
                        
                        # Create a shape object for each object
                        shape = {
                            "label": category,
                            "points": points,
                            "group_id": None,
                            "shape_type": "polygon",
                            "flags": {
                                "difficulty": difficulty
                            }
                        }
                        json_data["shapes"].append(shape)
            
            # Save the JSON file
            output_file = os.path.join(output_path, f"{image_id}.json")
            with open(output_file, "w") as f:
                json.dump(json_data, f, indent=2)

# Example usage
dota_path = "/home/tauproj5/venv/blog/detectron2/train/labelTxt-v1.0/labelTxt"
output_path = "/home/tauproj5/venv/blog/detectron2/train"
convert_dota_to_json(dota_path, output_path)
