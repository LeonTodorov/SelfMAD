from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class FaceParser():
    def __init__(self, device):
        self.device = device
        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.model.to(device)
        self.model.eval()

    def parse(self, img):
        # expects a PIL.Image or torch.Tensor
        inputs = self.image_processor(images=img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)
        upsampled_logits = nn.functional.interpolate(logits,
                size=img.size[::-1], # H x W
                mode='bilinear',
                align_corners=False)
        labels = upsampled_logits.argmax(dim=1)[0]
        labels_viz = labels.cpu().numpy()
        return labels_viz