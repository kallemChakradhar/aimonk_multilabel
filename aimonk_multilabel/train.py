import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import MultiLabelDataset
from model import get_model
from loss import MaskedBCELoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Paths (Change if needed)
IMG_DIR = "/content/drive/MyDrive/aimonk_multilabel/images"
LABEL_FILE = "/content/drive/MyDrive/aimonk_multilabel/labels.txt"

dataset = MultiLabelDataset(IMG_DIR, LABEL_FILE)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Model
model = get_model().to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Loss
criterion = MaskedBCELoss()

epochs = 10
loss_history = []
iteration = 0

model.train()

for epoch in range(epochs):
    for images, labels, mask in loader:
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        iteration += 1

        print(f"Epoch {epoch} Iter {iteration} Loss {loss.item():.4f}")

#  Save model
os.makedirs("/content/drive/MyDrive/aimonk_multilabel/weights", exist_ok=True)
torch.save(model.state_dict(), "/content/drive/MyDrive/aimonk_multilabel/weights/model.pth")

#  Plot loss curve
os.makedirs("/content/drive/MyDrive/aimonk_multilabel/plots", exist_ok=True)

plt.plot(loss_history)
plt.xlabel("iteration_number")
plt.ylabel("training_loss")
plt.title("Aimonk_multilabel_problem")
plt.savefig("/content/drive/MyDrive/aimonk_multilabel/plots/loss_curve.png")
plt.show()
