import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import create_dataloaders
from model import build_model, set_train_mode, get_device


MODULE_NAMES = [
    "conv1", "bn1", "layer1", "layer2",
    "layer3", "layer4", "fc",
]


def snapshot_bn(model):
    """Зберігає running_mean і running_var усіх BatchNorm2d."""
    result = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            result[name] = {
                "running_mean": module.running_mean.detach().clone(),
                "running_var": module.running_var.detach().clone(),
            }

    return result


def compare_bn(model, before):
    """Порівнює поточні статистики BatchNorm зі знімком."""
    changes = {}

    for name, module in model.named_modules():
        if name not in before:
            continue

        mean_changed = not torch.equal(
            before[name]["running_mean"],
            module.running_mean,
        )
        var_changed = not torch.equal(
            before[name]["running_var"],
            module.running_var,
        )

        changes[name] = {
            "mean": mean_changed,
            "var": var_changed,
        }

    return changes


def clone_state_dict(module):
    return {
        name: tensor.detach().clone()
        for name, tensor in module.state_dict().items()
    }


def state_changes(module, before):
    after = module.state_dict()

    return [
        name
        for name, old_value in before.items()
        if not torch.equal(old_value, after[name])
    ]


def run_five_steps(model, optimizer, loader, device, mode):
    """Виконує п'ять кроків оптимізації на перших п'яти batch."""
    if mode is None:
        # Навмисно неправильний режим для експерименту з BatchNorm
        model.train()
    else:
        set_train_mode(model, mode)

    iterator = iter(loader)

    for step in range(5):
        images, labels, _ = next(iterator)
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        print(f"  Step {step + 1}/5, loss={loss.item():.5f}")


def check_initial_loss(train_loader, device):
    print("\n" + "=" * 60)
    print("1. ПОЧАТКОВА ВТРАТА L0")
    print("=" * 60)

    # Фіксуємо перший batch, щоб обидві моделі отримали однакові дані.
    images, labels, _ = next(iter(train_loader))
    images = images.to(device)
    labels = labels.to(device)

    for mode in ["scratch", "frozen"]:
        model, _ = build_model(mode, seed=1, device=device)
        set_train_mode(model, mode)

        with torch.no_grad():
            logits = model(images)
            loss = F.cross_entropy(logits, labels)

        print(f"{mode:8s}: L0 = {loss.item():.6f}")

        del model


def check_parameter_and_buffer_changes(train_loader, device):
    print("\n" + "=" * 60)
    print("2. ПЕРЕВІРКА ПАРАМЕТРІВ І BUFFERS")
    print("=" * 60)

    for mode in ["scratch", "frozen", "unfreeze"]:
        print(f"\nРежим: {mode}")

        model, optimizer = build_model(
            mode,
            seed=1,
            device=device,
        )

        before = {
            name: clone_state_dict(getattr(model, name))
            for name in MODULE_NAMES
        }
        bn_before = snapshot_bn(model)

        run_five_steps(
            model, optimizer, train_loader, device, mode
        )

        print("\nЗміни стану модулів:")

        for name in MODULE_NAMES:
            changed = state_changes(
                getattr(model, name),
                before[name],
            )
            status = "ЗМІНИВСЯ" if changed else "без змін"
            print(f"  {name:8s}: {status}")

        print("\nЗміни всіх BatchNorm:")

        bn_changes = compare_bn(model, bn_before)

        for name, changes in bn_changes.items():
            mean_status = "змінився" if changes["mean"] else "без змін"
            var_status = "змінилася" if changes["var"] else "без змін"

            print(
                f"  {name}: "
                f"running_mean — {mean_status}; "
                f"running_var — {var_status}"
            )

        del model, optimizer


def check_batchnorm_bug(train_loader, device):
    print("\n" + "=" * 60)
    print("3. ЕКСПЕРИМЕНТ ІЗ model.train()")
    print("=" * 60)

    model, optimizer = build_model(
        "frozen",
        seed=1,
        device=device,
    )

    target_bn = model.layer1[0].bn1

    mean_before = target_bn.running_mean.detach().clone()
    var_before = target_bn.running_var.detach().clone()

    # Звичайний model.train() замість set_train_mode()
    run_five_steps(
        model,
        optimizer,
        train_loader,
        device,
        mode=None,
    )

    mean_diff = (
        target_bn.running_mean - mean_before
    ).abs().max().item()

    var_diff = (
        target_bn.running_var - var_before
    ).abs().max().item()

    print("\nРезультат для layer1.0.bn1:")
    print(f"Max |running_mean change|: {mean_diff:.10f}")
    print(f"Max |running_var change|:  {var_diff:.10f}")

    print("\nОчікування:")
    print("- Параметри backbone не змінюються.")
    print("- Статистики BatchNorm backbone змінюються.")

    del model, optimizer


def main():
    device = get_device()
    print("Device:", device)

    train_loader, _, _ = create_dataloaders()

    check_initial_loss(train_loader, device)

    check_parameter_and_buffer_changes(
        train_loader,
        device,
    )

    check_batchnorm_bug(train_loader, device)


if __name__ == "__main__":
    main()