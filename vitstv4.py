"""  Note -----------
This is the 4th version of ViTST.

In this version, CLS token output is used to calculate the style loss.
"""

import torch
from torchvision.utils import save_image
from utils import load_img, ViTGetFea
from einops import rearrange

# Set Device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Load Images
content_img = load_img('img/town.jpg')
style_img = load_img('img/starry_night.jpg')
generated_img = content_img.clone().requires_grad_(True)
# generated_img =torch.randn([1, 3, 224, 224], device=device, requires_grad=True)

# Set Hyper Parameters
total_step = 2001
learning_rate = 2e-2
alpha = 1  # Content Weight
beta = 1e7  # Style Weight

# Set Optimizer
optimizer = torch.optim.Adam([generated_img], lr=learning_rate)

# Create Feature Capture Model
get_feature = ViTGetFea([0, 1, 2])

# Painting
for step in range(total_step):

    content_loss = 0
    style_loss = 0

    # Create 2 Feature Maps <-- Shape: 1 x 197 x 768
    style_content, style_style = get_feature(style_img)
    content_content, content_style = get_feature(content_img)
    gen_content, gen_style = get_feature(generated_img)

    # Compute Content Loss
    for c, g in zip(content_content, gen_content):
        c = c.to(device)
        g = g.to(device)
        content_loss += torch.nn.functional.mse_loss(c, g)

    # Compute Style Loss
    for s, g in zip(style_style, gen_style):
        s = s[:, 0].to(device)
        g = g[:, 0].to(device)
        style_loss += torch.nn.functional.mse_loss(s, g)

    total_loss = alpha * content_loss + beta * style_loss

    # Update Value
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(generated_img[0, 0, 112, 100:110])
    
    if (step + 1) % 100 == 0:
        print(style_loss)
        save_image(generated_img, f'generated{step}.jpg')
        break
