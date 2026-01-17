import torch.nn as nn
from torchvision import transforms
from pytorch_pretrained_vit import ViT
from PIL import Image

def image_loader(image_name):
    imsize = 384
    loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()])

    image = Image.open(image_name).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image

class ViTFeatureExtractor(nn.Module):
    def __init__(self, chosen_layers):
        super().__init__()
        self.model = ViT('B_16_imagenet1k', pretrained=True).eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.features = {}
        self.chosen_layers = chosen_layers
        self._register_hooks()

    def _register_hooks(self):
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        for layer_idx in self.chosen_layers:
            # layer = self.model.transformer.blocks[layer_idx]
            layer = self.model.transformer.blocks[layer_idx].norm1
            layer.register_forward_hook(get_activation(f'block_{layer_idx}'))

    def forward(self, x):
        self.features = {}
        self.model(x)
        return self.features