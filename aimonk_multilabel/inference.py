import torch
from PIL import Image
import torchvision.transforms as transforms

from model import get_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = get_model()
model.load_state_dict(torch.load("/content/drive/MyDrive/aimonk_multilabel/weights/model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

attr_names = ["Attr1", "Attr2", "Attr3", "Attr4"]


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs)[0]

    result = [
        attr_names[i]
        for i, p in enumerate(probs)
        if p > 0.5
    ]

    print("Attributes present:", result)


if __name__ == "__main__":
    predict("/content/drive/MyDrive/aimonk_multilabel/images/image_0.jpg")   # change to any image path
