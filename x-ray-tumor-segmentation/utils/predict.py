import torch
import cv2
import matplotlib.pyplot as plt
from model.unet import UNet

# Load model
model = UNet()
model.load_state_dict(torch.load("xray_model.pth"))
model.eval()

# Load image
img = cv2.imread("test.jpg", 0)
img_resized = cv2.resize(img, (256, 256)) / 255.0

input_tensor = torch.tensor(img_resized).unsqueeze(0).unsqueeze(0).float()

# Predict
with torch.no_grad():
    output = model(input_tensor)

mask = output.squeeze().numpy()

# Show result
plt.subplot(1,2,1)
plt.title("X-ray")
plt.imshow(img, cmap='gray')

plt.subplot(1,2,2)
plt.title("Tumor Segmentation")
plt.imshow(mask, cmap='gray')

plt.show()