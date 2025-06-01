import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def build_model_object_detection(backbone='resnet50', num_class=2, use_pretrained=True):
    # 1. Create the same backbone used in the pretrained model
    # This will create resnet50 with FPN (Feature Pyramid Network)
    backbone = resnet_fpn_backbone('resnet50', pretrained=True)

    # 2. Define AnchorGenerator (same as default)
    # Correct the aspect_ratios to match the default pretrained model's RPN head expectation.
    # The default config uses 3 aspect ratios per spatial location.
    anchor_generator = AnchorGenerator(
      sizes=((32,), (64,), (128,), (256,), (512,)),
      aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    # 3. Define ROI Pooler (same as in torchvision)
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
      featmap_names=['0', '1', '2', '3'], # The default resnet_fpn_backbone returns 5 feature maps (p2, p3, p4, p5, p6).
                                              # Check the return_layers of the default backbone.
                                              # Looking at resnet_fpn_backbone source, it returns {"0": p2, "1": p3, "2": p4, "3": p5, "4": p6}.
                                              # So we need 5 feature map names.
      output_size=7,
      sampling_ratio=2
    )

    # 4. Assemble the Faster R-CNN model
    model = FasterRCNN(
      backbone=backbone,
      num_classes=num_class,  # use 2 for our case, car and background
      rpn_anchor_generator=anchor_generator,
      box_roi_pool=roi_pooler
    )

    if use_pretrained:

        pretrained_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

        pretrained_dict = pretrained_model.state_dict()
        model_dict = model.state_dict()

        #Filtered the last layer, because the difference number of class
        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and not k.startswith('roi_heads.box_predictor')
        }
        model.load_state_dict(filtered_dict, strict=False)

    return model