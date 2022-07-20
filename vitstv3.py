"""  Note -----------
This is the 3rd version of ViTST.

In this version, Gram Matrix is simply replaced by the output from the attention function.

As a result, there will be no content representation.
"""

import torch
from torchvision.utils import save_image
from utils import load_img, ViTGetFea
from einops import rearrange

# Set Device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Load Images
generated_img = load_img('img/town.jpg').requires_grad_(True)
style_img = load_img('img/starry_night.jpg')
# generated_img =torch.randn([1, 3, 224, 224], device=device, requires_grad=True)

# Set Hyper Parameters
total_step = 2001
learning_rate = 5e-3

# Set Optimizer
optimizer = torch.optim.Adam([generated_img], lr=learning_rate)

# Create Feature Capture Model
get_style = ViTGetFea([1])

# Painting
for step in range(total_step):

    style_loss = 0

    # Create 2 Feature Maps <-- Shape: 1 x 197 x 768
    style_features, _ = get_style(style_img)
    gen_features, _ = get_style(generated_img)

    # Compute Content Loss
    for s, g in zip(style_features, gen_features):
        s = s.to(device)
        g = g.to(device)
        style_loss += torch.nn.functional.mse_loss(s, g)

    # Update Value
    optimizer.zero_grad()
    style_loss.backward()
    optimizer.step()

    print(generated_img[0, 0, 112, 100:110])
    
    if (step + 1) % 200 == 0:
        print(style_loss)
        save_image(generated_img, f'generated{step}.jpg')
        break
