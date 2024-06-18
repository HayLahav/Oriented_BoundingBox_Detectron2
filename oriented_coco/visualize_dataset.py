import click
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from utils import get_labelme_dataset_function
import os

@click.command()
@click.argument("directory", nargs=1)
def main(directory):
    class_labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    dataset_name = "COCO_dataset"
    dataset_function = get_labelme_dataset_function(directory, class_labels)
    MetadataCatalog.get(dataset_name).thing_classes = class_labels
    DatasetCatalog.register(dataset_name, dataset_function)
    metadata = MetadataCatalog.get(dataset_name)
    for d in dataset_function():
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("Ground truth", out.get_image()[:, :, ::-1])
        cv2.waitKey(1000)

if __name__ == "__main__":
    main()
    
    

