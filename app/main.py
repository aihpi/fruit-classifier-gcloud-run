from fastapi import FastAPI, Depends, UploadFile, File
from pydantic import BaseModel
from torchvision import transforms
from torchvision.models import ResNet
from PIL import Image
import io
import torch
import torch.nn.functional as F

from app.model import load_model, load_transforms, CATEGORIES


class Result(BaseModel):
    category: str
    confidence: float


app = FastAPI()


@app.post("/predict", response_model=Result)
async def predict(
        input_image: UploadFile = File(...),
        model: ResNet = Depends(load_model),
        transforms: transforms.Compose = Depends(load_transforms)
) -> Result:
    # Read the uploaded image
    image = Image.open(io.BytesIO(await input_image.read()))

    # Convert RGBA image to RGB image
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Apply the transformations
    image = transforms(image).unsqueeze(0)  # Add batch dimension

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)

    # Map the predicted class index to the category
    category = CATEGORIES[predicted_class.item()]

    return Result(category=category, confidence=confidence.item())