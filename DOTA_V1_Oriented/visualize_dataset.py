import click
import cv2
import os
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from utils import get_labelme_dataset_function

@click.command()
@click.argument('directory', nargs=1)
def main(directory):
    dota_classes = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
                    "basketball-court", "ground-track-field", "harbor", "bridge",
                    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
                    "soccer-ball-field", "swimming-pool"]
    dataset_name = "dota_v1_dataset"
    dataset_function = get_labelme_dataset_function(directory, dota_classes)
    DatasetCatalog.register(dataset_name, dataset_function)
    MetadataCatalog.get(dataset_name).set(thing_classes=dota_classes)
    metadata = MetadataCatalog.get(dataset_name)

    dataset_dicts = dataset_function()

    for d in dataset_dicts:
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("DOTA v1 Dataset", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
