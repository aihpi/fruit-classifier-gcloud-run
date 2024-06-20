import os

import torch
import wandb
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet

MODELS_DIR = "models"
MODEL_FILE_NAME = "model.pth"

CATEGORIES = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]


def download_artifact():
    assert "WANDB_API_KEY" in os.environ, "Please enter the required environment variables."

    wandb.login()
    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")

    artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"
    artifact = wandb.Api().artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)


def get_raw_model() -> ResNet:
    # overwrite final classifier layer with our own output layers
    N_CLASSES = 6

    model = resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES)
    )

    return model


def load_model() -> ResNet:
    download_artifact()

    model = get_raw_model()
    model_state_dict_path = os.path.join(MODELS_DIR, MODEL_FILE_NAME)
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    return model


def load_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
