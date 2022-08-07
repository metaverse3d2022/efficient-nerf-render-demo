import matplotlib.pyplot as plt
import torch

tensors_model = torch.jit.load("tensors.pt")
img_rgb = list(tensors_model.parameters())[0]

plt.imshow(img_rgb)
plt.savefig("lego_rgb.png")