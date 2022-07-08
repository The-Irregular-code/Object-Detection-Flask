# import torchvision
# from torchvision.models.detection.faster_rcnn import AnchorGenerator
# from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


# def create_model(num_classes):

#     backbone = resnet_fpn_backbone(
#         'resnet50', pretrained=True, trainable_layers=5)

#     anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (
#         256,)), aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)]))

#     roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
#                                                     output_size=7, sampling_ratio=2)

#     # put the pieces together inside a FasterRCNN model
#     model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone, num_classes,
#                                                                 rpn_anchor_generator=anchor_generator,
#                                                                 box_roi_pool=roi_pooler)
#     return model


import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes):

    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# def create_model(num_classes):
#     model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
#         pretrained=True)
#     return model
