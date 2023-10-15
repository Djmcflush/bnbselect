### Segmentation model class
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


def caption_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    model.to(device)
    return model, processor
