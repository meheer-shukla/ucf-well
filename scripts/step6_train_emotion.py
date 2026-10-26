"""Step 6 — Train EfficientNet-B0 on AffectNet 3-class valence.

Works on both CPU and GPU (auto-detects). On a GTX 1650 Ti (4 GB VRAM),
finetuning EfficientNet-B0 at 128x128 takes ~5-8 min for 25 epochs.

Dataset: datasets/affectnet/YOLO_format/{train,valid,test}/
Classes: positive / neutral / negative

Run:
  python scripts/step6_train_emotion.py                    # default 128px, batch 16
  python scripts/step6_train_emotion.py --batch 8          # if OOM on 4 GB GPU
  python scripts/step6_train_emotion.py --img_size 160 --batch 12
  python scripts/step6_train_emotion.py --device cpu       # force CPU

Outputs:
  models/emotion_efficientnet_b0.pt      best model checkpoint
  cache/emotion_results.json             test metrics
  runs/emotion/emotion_history.json      per-epoch train/val history
  runs/emotion/training_curves.png
  runs/emotion/confusion_matrix.png
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    confusion_matrix, cohen_kappa_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.affectnet import AffectNetValence, VALENCE_NAMES


# ======================================================================
# Device detection
# ======================================================================
def pick_device(pref: str = "auto") -> str:
    """Return 'cuda' or 'cpu' with a clear reason."""
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] --device cuda requested but CUDA not available; "
                  "falling back to CPU")
            return "cpu"
        return "cuda"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device(device: str) -> None:
    print(f"PyTorch       : {torch.__version__}")
    print(f"Device        : {device}")
    if device == "cuda":
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        print(f"GPU           : {props.name}")
        print(f"VRAM          : {total_gb:.2f} GB")
        print(f"CUDA cap.     : {props.major}.{props.minor}")
        torch.backends.cudnn.benchmark = True
    else:
        if torch.cuda.is_available():
            print("[note] CUDA available but not used")
        else:
            print("[note] CUDA unavailable — running on CPU")


# ======================================================================
# Model
# ======================================================================
def build_model(n_classes: int = 3, freeze_backbone: bool = True):
    import torchvision.models as tvm
    from torchvision.models import EfficientNet_B0_Weights

    model = tvm.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("classifier"):
                p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes),
    )
    return model


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


# ======================================================================
# Train / eval loops
# ======================================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues, probas = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
        preds.append(proba.argmax(1))
        trues.append(y.cpu().numpy())
        probas.append(proba)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    probas = np.concatenate(probas)
    return (total_loss / len(trues), preds, trues, probas)


# ======================================================================
# OOM-safe batch-size probe
# ======================================================================
def fit_batch_size_to_memory(model, criterion, device, train_ds,
                             initial_bs: int) -> int:
    """Try one forward+backward with `initial_bs`. If OOM, halve and retry.

    Returns the batch size that succeeded.
    """
    if device != "cuda":
        return initial_bs

    bs = initial_bs
    while bs >= 2:
        try:
            dummy = torch.randn(bs, 3, 128, 128, device=device)
            dummy_y = torch.randint(0, 3, (bs,), device=device)
            logits = model(dummy)
            loss = criterion(logits, dummy_y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            del dummy, dummy_y, logits, loss
            torch.cuda.empty_cache()
            return bs
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            bs //= 2
            print(f"  [OOM] halving batch size → {bs}")
    raise RuntimeError("batch size 2 still OOM — GPU too small for this model")


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/affectnet")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16,
                    help="initial batch size; auto-halves on OOM")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_full", type=float, default=3e-4)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="0 is safest on Windows")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_model", default="models/emotion_efficientnet_b0.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device)

    print("=" * 72)
    print("EMOTION TRAINING — AffectNet 3-class valence")
    print("=" * 72)
    describe_device(device)
    print(f"Image size    : {args.img_size}")
    print(f"Batch size    : {args.batch} (will auto-halve on OOM)")
    print(f"Epochs        : {args.epochs}")
    print(f"Seed          : {args.seed}")
    print()

    os.makedirs("models", exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    os.makedirs("runs/emotion", exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds = AffectNetValence(args.root, "train",
                                img_size=args.img_size, augment=True)
    val_ds   = AffectNetValence(args.root, "valid",
                                img_size=args.img_size, augment=False)
    test_ds  = AffectNetValence(args.root, "test",
                                img_size=args.img_size, augment=False)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"Train class counts (pos/neu/neg): {train_ds.class_counts()}")
    print()

    # ------------------------------------------------------------------
    # Class weights
    # ------------------------------------------------------------------
    counts = np.array(train_ds.class_counts(), dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    print(f"Class weights (pos/neu/neg): {weights.round(3).tolist()}")
    print()
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(n_classes=3, freeze_backbone=True).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Model: EfficientNet-B0 ({n_trainable:,} trainable / "
          f"{n_total:,} total params)")

    # ------------------------------------------------------------------
    # OOM probe — pick a batch size that actually fits
    # ------------------------------------------------------------------
    fitted_bs = fit_batch_size_to_memory(model, criterion, device,
                                         train_ds, args.batch)
    if fitted_bs != args.batch:
        print(f"  [batch] using {fitted_bs} (reduced from {args.batch})")
    args.batch = fitted_bs
    print()

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=(device == "cuda"))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "val_f1": [],
               "epoch_seconds": [], "gpu_mem_mb": []}

    best_val_f1 = -1.0
    best_epoch = -1
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        if epoch == args.warmup_epochs + 1:
            unfreeze_all(model)
            n_trainable = sum(p.numel() for p in model.parameters()
                              if p.requires_grad)
            print(f"  [epoch {epoch}] unfroze full model "
                  f"({n_trainable:,} trainable params)")

        if epoch <= args.warmup_epochs:
            lr = args.lr_head
        else:
            progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
            lr = args.lr_full * 0.5 * (1 + np.cos(np.pi * progress))

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4,
        )

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device)
        va_loss, va_pred, va_true, _ = evaluate(model, val_loader,
                                                criterion, device)
        va_acc = accuracy_score(va_true, va_pred)
        va_f1  = f1_score(va_true, va_pred, average="macro", zero_division=0)
        dt = time.time() - t0

        gpu_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)
                   if device == "cuda" else 0.0)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["epoch_seconds"].append(dt)
        history["gpu_mem_mb"].append(gpu_mem)

        marker = ""
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_epoch = epoch
            torch.save({
                "state_dict": model.state_dict(),
                "img_size": args.img_size,
                "classes": VALENCE_NAMES,
                "val_f1": va_f1,
                "val_acc": va_acc,
                "epoch": epoch,
            }, args.out_model)
            marker = "  [saved]"
            patience_ctr = 0
        else:
            patience_ctr += 1

        mem_str = f"  gpu={gpu_mem:.0f}MB" if device == "cuda" else ""
        print(f"  ep {epoch:2d}/{args.epochs}  lr={lr:.2e}  "
              f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
              f"va_loss={va_loss:.4f}  va_acc={va_acc:.4f}  "
              f"va_f1={va_f1:.4f}  ({dt:.0f}s){mem_str}{marker}")

        if patience_ctr >= args.patience:
            print(f"  [early stop] no improvement for {args.patience} epochs")
            break

    print(f"\nBest val macro-F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    print("\nLoading best model for test evaluation...")
    ckpt = torch.load(args.out_model, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, te_pred, te_true, te_proba = evaluate(model, test_loader,
                                             criterion, device)

    test_metrics = {
        "accuracy": float(accuracy_score(te_true, te_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(te_true, te_pred)),
        "macro_f1": float(f1_score(te_true, te_pred, average="macro",
                                   zero_division=0)),
        "weighted_f1": float(f1_score(te_true, te_pred, average="weighted",
                                      zero_division=0)),
        "kappa": float(cohen_kappa_score(te_true, te_pred)),
        "per_class_f1": f1_score(te_true, te_pred, average=None,
                                 labels=[0, 1, 2],
                                 zero_division=0).tolist(),
        "confusion": confusion_matrix(te_true, te_pred,
                                      labels=[0, 1, 2]).tolist(),
        "class_names": VALENCE_NAMES,
        "n_test": int(len(te_true)),
        "device": device,
        "batch_size": int(args.batch),
    }

    print("\n" + "=" * 72)
    print("TEST RESULTS — AffectNet 3-class valence")
    print("=" * 72)
    print(f"  n             : {test_metrics['n_test']}")
    print(f"  accuracy      : {test_metrics['accuracy']:.4f}")
    print(f"  balanced acc  : {test_metrics['balanced_accuracy']:.4f}")
    print(f"  macro-F1      : {test_metrics['macro_f1']:.4f}")
    print(f"  weighted F1   : {test_metrics['weighted_f1']:.4f}")
    print(f"  Cohen kappa   : {test_metrics['kappa']:.4f}")
    print(f"  per-class F1  : "
          f"pos={test_metrics['per_class_f1'][0]:.4f}  "
          f"neu={test_metrics['per_class_f1'][1]:.4f}  "
          f"neg={test_metrics['per_class_f1'][2]:.4f}")
    print(f"  confusion     :")
    for i, name in enumerate(VALENCE_NAMES):
        print(f"    {name:8s}: {test_metrics['confusion'][i]}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open("cache/emotion_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open("runs/emotion/emotion_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[ OK ] wrote cache/emotion_results.json")
    print(f"[ OK ] wrote runs/emotion/emotion_history.json")
    print(f"[ OK ] wrote {args.out_model}")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ep = np.arange(1, len(history["train_loss"]) + 1)
    ax[0].plot(ep, history["train_loss"], label="train")
    ax[0].plot(ep, history["val_loss"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
    ax[0].set_title("Emotion training — loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(ep, history["train_acc"], label="train")
    ax[1].plot(ep, history["val_acc"], label="val acc")
    ax[1].plot(ep, history["val_f1"], label="val macro-F1")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("score")
    ax[1].set_title("Emotion training — metrics")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("runs/emotion/training_curves.png", dpi=110)
    plt.close()

    cm = np.array(test_metrics["confusion"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(VALENCE_NAMES); ax.set_yticklabels(VALENCE_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Test confusion — acc={test_metrics['accuracy']:.3f}, "
                 f"F1={test_metrics['macro_f1']:.3f}")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("runs/emotion/confusion_matrix.png", dpi=110)
    plt.close()
    print(f"[ OK ] wrote runs/emotion/training_curves.png")
    print(f"[ OK ] wrote runs/emotion/confusion_matrix.png")


if __name__ == "__main__":
    main()