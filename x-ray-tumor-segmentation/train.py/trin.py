import torch
from torch.utils.data import DataLoader
from model.unet import UNet
from utils.dataset import XrayDataset

# Load dataset
dataset = XrayDataset("data/images", "data/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model
model = UNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCELoss()

# Training loop
for epoch in range(10):
    for images, masks in loader:
        preds = model(images)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "xray_model.pth")