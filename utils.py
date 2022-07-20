import torch
import torch.nn as nn
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pytorch_pretrained_vit import ViT
from PIL import Image

# Set Device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Define Data Loader
transform = Compose([Resize((224, 224)), ToTensor()])


def load_img(img_name):
    data = Image.open(img_name)
    data = transform(data)
    data = data.unsqueeze(0)
    return data.to(device)


# Define a Hook Class (New Version)
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


# Define a Hook for CLS Token
class SaveOutput1:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


# Define a Feature Container (Old Version)
def ViTST(model_list, data):
    features = []
    for model in model_list:
        x = model(data)
        features.append(x)
    return features


# Define a ViT Class
class ViTGetFea(nn.Module):
    def __init__(self, chosen_features):
        super(ViTGetFea, self).__init__()

        self.chosen_features = chosen_features
        self.save_output = SaveOutput()
        self.save_output1 = SaveOutput1()
        self.model = ViT('B_16', pretrained=True).to(device).eval()
        del self.model.positional_embedding
        self.handle_list = []

    def forward(self, x):

        for i in self.chosen_features:
            handle = self.model.transformer.blocks[i].attn.register_forward_hook(self.save_output)
            self.handle_list.append(handle)
        handle1 = self.model.norm.register_forward_pre_hook(self.save_output1)

        self.save_output1.clear()
        self.save_output.clear()
        self.model(x)
        self.features = self.save_output.outputs
        self.features1 = self.save_output1.outputs

        for i in self.handle_list:
            i.remove()
        handle1.remove()

        return self.features, self.features1[0][0]
