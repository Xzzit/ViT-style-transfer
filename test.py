import torch
from pytorch_pretrained_vit import ViT
from utils import load_img, SaveOutput, ViTGetFea

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Print the architecture of the model
model = ViT('B_16', pretrained=True).to(device).eval()
print(model)

# Print ViTGetFea Function
img = load_img('img/town.jpg')
get_feature = ViTGetFea([0, 1, 2])
print("Length of ViTGetFea: ", len(get_feature(img)))
print("Tensor 1 length: ", len(get_feature(img)[0]))
print("Tensor 1 shape: ", get_feature(img)[0][0].shape)
print("Tensor 2 shape: ", get_feature(img)[1].shape)
