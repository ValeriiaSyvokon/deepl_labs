
import torch

from dataset import create_dataloaders, CLASSES, IMG_SIZE
from model import build_model, set_train_mode
from train import train_one_epoch, evaluate

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs" / "unfreeze_seed2"

EPOCHS = 8
SEED = 2
MODE = "unfreeze"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = create_dataloaders()

    model, optimizer = build_model(
        mode=MODE,
        seed=SEED,
        device=device,
    )

    print("Повторне навчання вибраної моделі")
    print("Режим:", MODE)
    print("Seed:", SEED)
    print("Пристрій:", device)

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            MODE,
        )

        val_f1, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Loss: {loss:.6f} | "
            f"Val macro-F1: {val_f1:.6f}"
        )

    # Зберігаємо ваги після восьмої епохи
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        model.state_dict(),
        RUN_DIR / "model_state.pt",
    )

    print("\nВаги збережено:", RUN_DIR / "model_state.pt")

    checkpoint = {
        "state_dict": model.state_dict(),
        "regime": MODE,
        "seed": SEED,
        "classes": CLASSES,
        "img_size": IMG_SIZE,
    }

    torch.save(checkpoint, ROOT / "lab3_model.pt")

    print(f"Фінальну модель збережено: {ROOT / 'lab3_model.pt'}")

if __name__ == "__main__":
    main()
