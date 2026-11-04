import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights


NUM_CLASSES = 6

LR_FC = 3e-3
LR_LAYER4 = 1e-4
WEIGHT_DECAY = 1e-4


def build_model(mode: str, seed: int, device):
    """
    Створює ResNet-18 для одного з трьох режимів:
    scratch, frozen, unfreeze.
    """

    if mode not in {"scratch", "frozen", "unfreeze"}:
        raise ValueError(f"Невідомий режим: {mode}")

    # 1. Фіксуємо генератори випадкових чисел
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Створюємо ResNet-18
    if mode == "scratch":
        weights = None
    else:
        weights = ResNet18_Weights.IMAGENET1K_V1

    model = resnet18(weights=weights)

    # 3. Замінюємо класифікатор на 6 класів
    model.fc = nn.Linear(512, NUM_CLASSES)

    # 4. Налаштовуємо trainable-параметри
    if mode == "scratch":
        for param in model.parameters():
            param.requires_grad = True

    elif mode == "frozen":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.fc.parameters():
            param.requires_grad = True

    elif mode == "unfreeze":
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    # 5. Переносимо модель на пристрій
    model = model.to(device)

    # 6. Створюємо оптимізатор лише для trainable-параметрів
    if mode == "scratch":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LR_FC,
            weight_decay=WEIGHT_DECAY,
        )

    elif mode == "frozen":
        optimizer = torch.optim.AdamW(
            model.fc.parameters(),
            lr=LR_FC,
            weight_decay=WEIGHT_DECAY,
        )

    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.fc.parameters(),
                    "lr": LR_FC,
                },
                {
                    "params": model.layer4.parameters(),
                    "lr": LR_LAYER4,
                },
            ],
            weight_decay=WEIGHT_DECAY,
        )

    return model, optimizer


def set_train_mode(model, mode: str):
    """
    Вмикає навчальний режим, залишаючи заморожені
    частини мережі в eval().
    """

    model.train()

    if mode == "frozen":
        frozen_modules = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        ]

    elif mode == "unfreeze":
        frozen_modules = [
            model.conv1,
            model.bn1,
            model.layer1,
            model.layer2,
            model.layer3,
        ]

    elif mode == "scratch":
        frozen_modules = []

    else:
        raise ValueError(f"Невідомий режим: {mode}")

    for module in frozen_modules:
        module.eval()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


if __name__ == "__main__":
    device = get_device()
    print("Device:", device)

    for mode in ["scratch", "frozen", "unfreeze"]:
        model, optimizer = build_model(mode, seed=1, device=device)

        total, trainable = count_parameters(model)

        print(f"\nMode: {mode}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

        print("Optimizer groups:")

        for i, group in enumerate(optimizer.param_groups):
            num_params = sum(
                p.numel() for p in group["params"]
            )

            print(
                f"  Group {i + 1}: "
                f"{num_params:,} parameters, "
                f"lr={group['lr']}"
            )