"""  Note -----------
This is the 2nd version of ViTST.

This file package the hook function into a class, which allows you to use it more conveniently.

Also, in this version, Gram Matrix is computed to represnet the style, which doesn't produce a decent result.

So in the next vesion, Gram Matrix will be simply replaced by the output from the attention function.
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
generated_img = load_img('img/town.jpg').requires_grad_(True)
# generated_img =torch.randn([1, 3, 224, 224], device=device, requires_grad=True)

# Set Hyper Parameters
total_step = 10000
learning_rate = 1e-2
alpha = 1  # Content Weight
beta = 1  # Style Weight

# Set Optimizer
optimizer = torch.optim.Adam([generated_img], lr=learning_rate)

# Create Feature Capture Model
get_content = ViTGetFea([9, 10, 11])
get_style = ViTGetFea([6, 7, 8])

# Painting
for step in range(total_step):

    content_loss = style_loss = 0

    # Create 4 Feature Maps(2 for Generated Img) <-- Shape: 1 x 197 x 768
    content_features, _ = get_content(content_img)
    style_features, _ = get_style(style_img)
    generated_content_features, _ = get_content(generated_img)
    generated_style_features, _ = get_style(generated_img)

    # Compute Content Loss
    for c, g in zip(content_features, generated_content_features):
        c = c.to(device)
        g = g.to(device)
        content_loss += torch.nn.functional.mse_loss(c, g)

    # Compute Style Loss
    for s, g in zip(style_features, generated_style_features):
        s = s.to(device)
        g = g.to(device)
        s = rearrange(s, 'b f (h d) -> b h (f d)', h=8)
        g = rearrange(g, 'b f (h d) -> b h (f d)', h=8)
        style_grammatrix = torch.einsum('bhd, bid->bhi', s, s)
        generated_grammatrix = torch.einsum('bhd, bid->bhi', g, g)
        style_loss += torch.nn.functional.mse_loss(generated_grammatrix, style_grammatrix)

    # Weight the Content & Style
    total_loss = alpha * content_loss + beta * style_loss

    # Update Value
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(generated_img[0, 0, 112, 100:110])

    if (step + 1) % 300 == 0:
        print(total_loss)
        save_image(generated_img, f'generated{step}.jpg')
        break
