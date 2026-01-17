import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image

from utils import image_loader, ViTFeatureExtractor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loss function
def gram_matrix(input):
    # input shape: (B, 197, 768)
    features = input[:, 1:, :] # drop the CLS token
    
    (b, n, c) = features.size()
    features = features.view(b, n, c)
    
    # compute Gram Matrix
    # (B, C, N) @ (B, N, C) -> (B, C, C)
    features = features.transpose(1, 2)
    G = torch.bmm(features, features.transpose(1, 2))
    
    return G

def total_variation_loss(img):
    return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

# load images
style_img = image_loader("img/starry_night.jpg").to(device)
content_img = image_loader("img/town.jpg").to(device)

input_img = content_img.clone()
input_img = nn.Parameter(input_img) 

# feature extractor
extractor = ViTFeatureExtractor(chosen_layers=[0, 2, 4, 7, 10]).to(device)

# optimizer
# optimizer = optim.Adam([input_img], lr=5e-3)
optimizer = optim.LBFGS([input_img])

print("Calculating targets...")
with torch.no_grad():
    style_features = extractor(style_img)
    content_features = extractor(content_img)
    
    style_grams = {k: gram_matrix(v) for k, v in style_features.items() if k in ['block_0', 'block_2', 'block_4']}
    content_targets = {k: v for k, v in content_features.items() if k in ['block_7', 'block_10']}

style_weight = 8e-1
content_weight = 1
tv_weight = 1e-3

print("Start Training...")
run = [0]
while run[0] <= 1000:
    def closure():
        with torch.no_grad():
            input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        features = extractor(input_img)
        
        style_score = 0
        content_score = 0

        # Style Loss
        for layer in ['block_0', 'block_2', 'block_4']:
            gm = gram_matrix(features[layer])
            style_score += F.mse_loss(gm, style_grams[layer])
        
        # Content Loss
        for layer in ['block_7', 'block_10']:
            content_score += F.mse_loss(features[layer], content_targets[layer])
            
        # TV Loss
        tv_score = total_variation_loss(input_img)

        loss = style_score * style_weight + content_score * content_weight + tv_score * tv_weight
        loss.backward()

        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Run {run[0]}: Style Loss : {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
            save_image(input_img, f'output/run_{run[0]}.jpg')

        return loss

    optimizer.step(closure)

with torch.no_grad():
    input_img.data.clamp_(0, 1)
    save_image(input_img, 'output/final.jpg')