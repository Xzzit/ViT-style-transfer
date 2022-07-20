"""  Note: ----------

This is the 1st version of Style Transfer using Vision Transformer.

This version uses hook instead of del model separately, which could allow you output any function in a specific layer.
"""


import torch
from torchvision.utils import save_image
from utils import load_img, SaveOutput
from pytorch_pretrained_vit import ViT
from einops import rearrange

# Set Device
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Load Images
content_img = load_img('img/town.jpg')
style_img = load_img('img/starry_night.jpg')
# generated_img = content_img.clone().requires_grad_(True)
generated_img =torch.randn([1, 3, 224, 224], device=device, requires_grad=True)

# Set Hyper Parameters
total_step = 10000
learning_rate = 1e-3
alpha = 1  # Content Weight
beta = 1e7  # Style Weight

# Set Optimizer
optimizer = torch.optim.Adam([generated_img], lr=learning_rate)

# Load Model
save_output = SaveOutput()
model = ViT('B_16', pretrained='True').to(device).eval()
model.requires_grad_(False)

# Create Hook for Every --> proj layer ^ attn layer ^ etc...
handle_list = []
for i in range(12):
    handle = model.transformer.blocks[i].pwff.register_forward_hook(save_output)
    handle_list.append(handle)

# Painting
for step in range(total_step):

    content_loss = style_loss = 0

    # Create 3 Feature Maps <-- Shape: 1 x 197 x 768
    model(content_img)
    content_features = [save_output.outputs[i] for i in [10, 11]]
    save_output.clear()

    model(style_img)
    style_features = [save_output.outputs[i] for i in [1, 2, 3, 4, 5, 6]]
    save_output.clear()

    model(generated_img)
    generated_features = [save_output.outputs[i] for i in [1, 2, 3, 4, 5, 6, 10, 11]]
    save_output.clear()

    # Compute Content Loss
    for c, g in zip(content_features, generated_features[6:]):
        c = c.to(device)
        g = g.to(device)
        content_loss += torch.nn.functional.mse_loss(c, g)
  
    # Compute Style Loss
    for s, g in zip(style_features, generated_features):
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
    
    if (step + 1) % 1000 == 0:
        print(total_loss)
        save_image(generated_img, f'generated{step}.jpg')
