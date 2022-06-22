from datasets.pre_process import get_transforms
from datasets.custom_datasets import ClassifiedDatasets, get_dataloaders
from models import get_model
from utils import test_model, train_model, convert_to_onnx
from datasets.export import export_classes

import torch
from torch.backends import cudnn

cudnn.benchmark = True

train_transform, eval_transform = get_transforms()

anime_datasets = ClassifiedDatasets(
    "data/anime-pictures",
    train_transform,
    eval_transform,
    validation_split=0.2,
)

export_classes(anime_datasets.classes, "characters.json")
train_loader, val_loader, test_loader = get_dataloaders(anime_datasets)

model = get_model(output_size=len(anime_datasets.classes))

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    exp_lr_scheduler,
    device=device,
)

test_model(model, test_loader)

torch.save(model.state_dict(), "anime.pth")
convert_to_onnx(model, "anime.onnx", input_names=["input"], output_names=["output"])
convert_to_onnx(model, "../anime.onnx", input_names=["input"], output_names=["output"])
