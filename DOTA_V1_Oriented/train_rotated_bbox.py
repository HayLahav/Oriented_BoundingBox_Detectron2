import click
import numpy as np
import os
import torch
from utils import get_labelme_dataset_function
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, detection_utils as utils, transforms as T, build_detection_train_loader
from detectron2.engine import launch, DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.config import get_cfg

def rotate_bbox(annotation, transforms):
    annotation["bbox"] = transforms.apply_rotated_box(
        np.asarray([annotation['bbox']]))[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS
    return annotation

def get_shape_augmentations():
    # Optional shape augmentations
    return [
        T.RandomFlip(),
        T.ResizeShortestEdge(short_edge_length=(
            640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice'),
        T.RandomFlip()
    ]

def get_color_augmentations():
    # Optional color augmentations
    return T.AugmentationList([
        T.RandomBrightness(0.9, 1.1),
        T.RandomSaturation(intensity_min=0.75, intensity_max=1.25),
        T.RandomContrast(intensity_min=0.76, intensity_max=1.25)
    ])

def dataset_mapper(dataset_dict):
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    color_aug_input = T.AugInput(image)
    get_color_augmentations()(color_aug_input)
    image = color_aug_input.image
    image, image_transforms = T.apply_transform_gens(
        get_shape_augmentations(), image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))
    annotations = [
        rotate_bbox(obj, image_transforms)
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances_rotated(
        annotations, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict

class RotatedBoundingBoxTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [RotatedCOCOEvaluator(
            dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=dataset_mapper)

def check_dataset_preparation(dataset_name):
    # Check if the dataset is registered in the DatasetCatalog
    if dataset_name not in DatasetCatalog:
        print(f"Dataset '{dataset_name}' is not registered in the DatasetCatalog.")
        return

    # Get the dataset from the DatasetCatalog
    dataset_dicts = DatasetCatalog.get(dataset_name)

    # Check if the dataset is empty
    if not dataset_dicts:
        print(f"Dataset '{dataset_name}' is empty.")
        return

    # Check if the dataset is in the expected format
    expected_keys = ["file_name", "height", "width", "image_id", "annotations"]
    for dataset_dict in dataset_dicts:
        if not all(key in dataset_dict for key in expected_keys):
            print(f"Dataset '{dataset_name}' has missing keys in the dataset dict.")
            return

    print(f"Dataset '{dataset_name}' is properly loaded and in the expected format.")

def train_detectron(directory, num_gpus):
    dota_classes = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
                    "basketball-court", "ground-track-field", "harbor", "bridge",
                    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
                    "soccer-ball-field", "swimming-pool"]
    dataset_function = get_labelme_dataset_function(directory, dota_classes)
    dataset_name = "dota_v1_dataset"
    DatasetCatalog.register(dataset_name, dataset_function)

    # Check the dataset preparation
    check_dataset_preparation(dataset_name)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # Base model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Weights
    # Rotated bbox specific config in the same directory as this file
    cfg.merge_from_file(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "rotated_bbox_config.yaml"))
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = (dataset_name,)
    # Directory where the checkpoints are saved, "." is the current working dir
    cfg.OUTPUT_DIR = "."
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(dota_classes)
    trainer = RotatedBoundingBoxTrainer(cfg)
    # NOTE: important, the model will not train without this
    trainer.resume_or_load(resume=False)
    trainer.train()

@click.command()
@click.argument('directory', nargs=1)
@click.option('--num-gpus', default=0, help='Number of GPUs to use, default zero')
def main(directory, num_gpus):
    launch(
        train_detectron,
        num_gpus,
        args=(directory, num_gpus),
    )
    

    

if __name__ == "__main__":
    main()
