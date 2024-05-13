import click
import cv2
import os
from utils import get_labelme_dataset_function
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


@click.command()
@click.argument('directory', nargs=1)
@click.option('--weights', default="model_final.pth", help='Path to the model to use')
def main(directory, weights):
    dota_classes = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
                    "basketball-court", "ground-track-field", "harbor", "bridge",
                    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
                    "soccer-ball-field", "swimming-pool"]
    dataset_name = "dota_v1_dataset"
    
    dataset_function = get_labelme_dataset_function(directory, dota_classes)
    MetadataCatalog.get(dataset_name).thing_classes = dota_classes
    DatasetCatalog.register(dataset_name, dataset_function)
    metadata = MetadataCatalog.get(dataset_name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Base model
    # Rotated bbox specific config in the same directory as this file
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dota_classes)
    
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.99  # Prediction confidence threshold

    predictor = DefaultPredictor(cfg)

    for d in dataset_function():
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        v = Visualizer(img[:, :, ::-1], metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("Predictions", out.get_image()[:, :, ::-1])
        cv2.imwrite("prediction.png", out.get_image()[:, :, ::-1])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
