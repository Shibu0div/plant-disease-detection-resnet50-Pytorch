from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import torch
import torch.nn as nn
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS = ['Apple_Black_Rot', 'Apple_Cedar_Rust', 'Apple_Healthy', 'Apple_Scab', 'Cherry_Healthy', 'Cherry_Powdery_Mildew', 'Corn_Cercospora_Leaf_Spot', 'Corn_Common_Rust', 'Corn_Healthy', 'Corn_Northern_Leaf_Blight', 'Grape_Black_Rot', 'Grape_Esca_Black_Measles', 'Grape_Healthy', 'Grape_Leaf_Blight', 'Peach_Bacterial_Spot', 'Peach_Healthy', 'Pepper_Bacterial_Spot', 'Pepper_Healthy', 'Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight', 'Strawberry_Healthy', 'Strawberry_Leaf_Scorch', 'Tomato_Bacterial_Spot', 'Tomato_Early_Blight', 'Tomato_Healthy', 'Tomato_Late_Blight', 'Tomato_Septoria_Leaf_Spot', 'Tomato_Yellow_Leaf_Curl_Virus']
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])
model_path = "saved_model/best_model_resNet50_Finetuned.pt"
def detect(file):
    image = Image.open(file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 29)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu"
        )
    )

    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

        probability = torch.softmax(output, dim=1)[0]
        confidence, pred = torch.max(probability, 0)

    # for i, prob in enumerate(probability):
    #     print(CLASS[i], prob.item())

    label = CLASS[pred.item()]

    return label, confidence.item()
