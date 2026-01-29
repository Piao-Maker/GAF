import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flamedataset import FlameDatasetFrom3Dirs
from resnet18 import MultiModalFlameNet

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# 画 confusion matrix 用
import matplotlib.pyplot as plt


@dataclass
class EvalResults:
    loss: float
    acc: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    per_class: dict
    cm: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray

    # ===== fire(正类=1) 相关 =====
    y_score_fire: np.ndarray
    roc_auc_fire: float
    pr_auc_fire: float
    fnr_fire: float
    fpr_fire: float


def _sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@torch.no_grad()
def evaluate(model, loader, device, criterion, class_names):
    """Evaluate on a loader.

    Assumes **binary classification** with labels:
        0 = smoke
        1 = fire (positive)

    Returns common metrics + ROC-AUC/PR-AUC for fire as positive class,
    and FNR/FPR (fire as positive).
    """

    model.eval()

    total_loss = 0.0
    total = 0

    all_labels = []
    all_preds = []
    all_scores_fire = []  # fire probability (positive score)

    for imgs, gafs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        gafs = gafs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs, gafs)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        # fire(正类=1) 的概率分数，用于 AUC/PR-AUC
        probs = torch.softmax(logits, dim=1)
        if probs.shape[1] >= 2:
            score_fire = probs[:, 1]
        else:
            # 极端情况：模型输出维度不对，避免崩溃
            score_fire = torch.zeros((probs.shape[0],), device=probs.device)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total += bs

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_scores_fire.extend(score_fire.detach().cpu().numpy().tolist())

    if total == 0:
        # 极端情况：空验证集
        y_true = np.array([], dtype=np.int64)
        y_pred = np.array([], dtype=np.int64)
        y_score_fire = np.array([], dtype=np.float32)
        cm = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        return EvalResults(
            loss=float("nan"),
            acc=float("nan"),
            precision_macro=float("nan"),
            recall_macro=float("nan"),
            f1_macro=float("nan"),
            precision_weighted=float("nan"),
            recall_weighted=float("nan"),
            f1_weighted=float("nan"),
            per_class={},
            cm=cm,
            y_true=y_true,
            y_pred=y_pred,
            y_score_fire=y_score_fire,
            roc_auc_fire=float("nan"),
            pr_auc_fire=float("nan"),
            fnr_fire=float("nan"),
            fpr_fire=float("nan"),
        )

    y_true = np.array(all_labels, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_score_fire = np.array(all_scores_fire, dtype=np.float32)

    val_loss = total_loss / total
    acc = accuracy_score(y_true, y_pred)

    labels_order = list(range(len(class_names)))

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, average="weighted", zero_division=0
    )

    prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_order, average=None, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(prec_cls[i]),
            "recall": float(rec_cls[i]),
            "f1": float(f1_cls[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels_order)

    # ===== fire(正类=1) 的 ROC-AUC / PR-AUC =====
    # 如果验证集只包含单一类别，sklearn 会抛 ValueError
    try:
        roc_auc_fire = float(roc_auc_score(y_true, y_score_fire))
    except ValueError:
        roc_auc_fire = float("nan")

    try:
        pr_auc_fire = float(average_precision_score(y_true, y_score_fire))
    except ValueError:
        pr_auc_fire = float("nan")

    # ===== fire(正类=1) 的 FNR / FPR =====
    # 二分类时 cm 应为 2x2：[[TN, FP],[FN, TP]]
    fnr_fire = float("nan")
    fpr_fire = float("nan")
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel().tolist()
        fnr_fire = (fn / (fn + tp)) if (fn + tp) > 0 else float("nan")
        fpr_fire = (fp / (fp + tn)) if (fp + tn) > 0 else float("nan")

    return EvalResults(
        loss=float(val_loss),
        acc=float(acc),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
        per_class=per_class,
        cm=cm,
        y_true=y_true,
        y_pred=y_pred,
        y_score_fire=y_score_fire,
        roc_auc_fire=roc_auc_fire,
        pr_auc_fire=pr_auc_fire,
        fnr_fire=float(fnr_fire),
        fpr_fire=float(fpr_fire),
    )


@torch.no_grad()
def benchmark_inference(model, loader, device, max_batches=50, warmup_batches=10):
    """Only benchmark forward time (no loss/backward).

    Returns:
        ms_per_img, fps

    Notes:
        - GPU timing uses torch.cuda.synchronize() for correctness.
        - Warmup is performed to stabilize kernels.
    """

    model.eval()

    # warmup
    it = iter(loader)
    for _ in range(warmup_batches):
        try:
            imgs, gafs, _ = next(it)
        except StopIteration:
            break
        imgs = imgs.to(device, non_blocking=True)
        gafs = gafs.to(device, non_blocking=True)
        _ = model(imgs, gafs)
    _sync_cuda()

    total_images = 0
    start = time.perf_counter()

    batches = 0
    for imgs, gafs, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        gafs = gafs.to(device, non_blocking=True)

        _sync_cuda()
        _ = model(imgs, gafs)
        _sync_cuda()

        total_images += imgs.size(0)
        batches += 1

        if batches >= max_batches:
            break

    end = time.perf_counter()
    elapsed = end - start

    if total_images == 0 or elapsed <= 0:
        return float("nan"), float("nan")

    sec_per_img = elapsed / total_images
    ms_per_img = sec_per_img * 1000.0
    fps = 1.0 / sec_per_img
    return ms_per_img, fps


def save_confusion_matrix_png(cm, class_names, out_path: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm)
    fig.colorbar(im)

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)

    # annotate values
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    # ====== 基本配置 ======
    root_dir = r"D:\BaiduNetdiskDownload\fire\data\ylp_data"  # 你自己的数据路径
    img_size = 128
    series_length = 64
    batch_size = 16
    num_workers = 4
    num_epochs = 20

    # 二分类：0=smoke, 1=fire
    class_names = ["smoke", "fire"]
    num_classes = 2

    run_dir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    metrics_txt = os.path.join(run_dir, "metrics.txt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] device = {device}")
    print(f"[Info] run_dir = {run_dir}")

    # ====== 数据集 ======
    train_dataset = FlameDatasetFrom3Dirs(
        root_dir=root_dir,
        split="train",
        img_size=img_size,
        series_length=series_length,
        gaf_method="summation",
        use_augmentation=True,
    )

    val_dataset = FlameDatasetFrom3Dirs(
        root_dir=root_dir,
        split="val",
        img_size=img_size,
        series_length=series_length,
        gaf_method="summation",
        use_augmentation=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ====== 模型 / 损失 / 优化器 ======
    model = MultiModalFlameNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ====== 训练 ======
    best_f1 = -1.0
    best_path = os.path.join(run_dir, "best.pt")

    train_start = time.perf_counter()

    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"device={device}\n")
        f.write(f"class_names={class_names}\n")
        f.write(f"img_size={img_size}, series_length={series_length}\n")
        f.write(f"batch_size={batch_size}, num_epochs={num_epochs}\n")
        f.write("fire-positive metrics: ROC-AUC/PR-AUC/FNR/FPR (positive label=1)\n\n")

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()

        # ---- train ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, gafs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            gafs = gafs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs, gafs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += bs

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # ---- val ----
        eval_res = evaluate(model, val_loader, device, criterion, class_names)

        epoch_time = time.perf_counter() - epoch_start

        # 输出简表
        print(
            f"[Epoch {epoch + 1:02d}/{num_epochs:02d}] "
            f"time={format_seconds(epoch_time)} "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
            f"val_loss={eval_res.loss:.6f} val_acc={eval_res.acc:.6f} "
            f"prec_macro={eval_res.precision_macro:.6f} rec_macro={eval_res.recall_macro:.6f} f1_macro={eval_res.f1_macro:.6f} "
            f"prec_w={eval_res.precision_weighted:.6f} rec_w={eval_res.recall_weighted:.6f} f1_w={eval_res.f1_weighted:.6f}"
        )

        print(
            f"  fire-positive: roc_auc={eval_res.roc_auc_fire:.6f} "
            f"pr_auc={eval_res.pr_auc_fire:.6f} "
            f"FNR_fire={eval_res.fnr_fire:.6f} "
            f"FPR_fire={eval_res.fpr_fire:.6f}"
        )

        # per-class
        print("  per-class:")
        for cname in class_names:
            m = eval_res.per_class.get(cname, {})
            if m:
                print(
                    f"    {cname:<5s}: "
                    f"P={m['precision']:.6f} R={m['recall']:.6f} F1={m['f1']:.6f} N={m['support']}"
                )

        # confusion matrix
        print("  confusion_matrix:")
        print(eval_res.cm)

        # classification report
        report = classification_report(
            eval_res.y_true,
            eval_res.y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
        print("  classification_report:")
        print(report)

        # 写文件记录
        with open(metrics_txt, "a", encoding="utf-8") as f:
            f.write(
                f"[Epoch {epoch + 1:02d}/{num_epochs:02d}] "
                f"time={format_seconds(epoch_time)} "
                f"train_loss={train_loss:.6f} train_acc={train_acc:.6f} "
                f"val_loss={eval_res.loss:.6f} val_acc={eval_res.acc:.6f} "
                f"prec_macro={eval_res.precision_macro:.6f} rec_macro={eval_res.recall_macro:.6f} f1_macro={eval_res.f1_macro:.6f} "
                f"prec_w={eval_res.precision_weighted:.6f} rec_w={eval_res.recall_weighted:.6f} f1_w={eval_res.f1_weighted:.6f} "
                f"roc_auc_fire={eval_res.roc_auc_fire:.6f} pr_auc_fire={eval_res.pr_auc_fire:.6f} "
                f"FNR_fire={eval_res.fnr_fire:.6f} FPR_fire={eval_res.fpr_fire:.6f}\n"
            )
            f.write("per-class:\n")
            for cname in class_names:
                m = eval_res.per_class.get(cname, {})
                if m:
                    f.write(f"  {cname}: {m}\n")
            f.write("confusion_matrix:\n")
            f.write(str(eval_res.cm) + "\n")
            f.write("classification_report:\n")
            f.write(report + "\n")
            f.write("-" * 80 + "\n")

        # 保存 best（仍以 macro-F1 作为选择标准）
        if eval_res.f1_macro > best_f1:
            best_f1 = eval_res.f1_macro
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_macro_f1": best_f1,
                    "class_names": class_names,
                },
                best_path,
            )
            print(f"[Info] Saved best checkpoint to: {best_path} (macroF1={best_f1:.6f})")

    total_train_time = time.perf_counter() - train_start
    print(f"[Done] Total training time: {format_seconds(total_train_time)}")

    # ====== 载入 best，再做一次最终评估 + 推理测速 ======
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(
            f"[Info] Loaded best checkpoint from epoch {ckpt.get('epoch')} "
            f"(macroF1={ckpt.get('best_macro_f1'):.6f})"
        )

    final_eval = evaluate(model, val_loader, device, criterion, class_names)
    cm_png = os.path.join(run_dir, "confusion_matrix.png")
    save_confusion_matrix_png(final_eval.cm, class_names, cm_png)
    print(f"[Info] Saved confusion matrix figure: {cm_png}")

    ms_per_img, fps = benchmark_inference(model, val_loader, device, max_batches=50, warmup_batches=10)
    print(f"[Infer] latency={ms_per_img:.3f} ms/img, FPS={fps:.2f}")

    with open(metrics_txt, "a", encoding="utf-8") as f:
        f.write("\nFINAL_EVAL:\n")
        f.write(f"val_loss={final_eval.loss:.6f}, val_acc={final_eval.acc:.6f}\n")
        f.write(
            f"macro: P={final_eval.precision_macro:.6f} R={final_eval.recall_macro:.6f} F1={final_eval.f1_macro:.6f}\n"
        )
        f.write(
            f"weighted: P={final_eval.precision_weighted:.6f} R={final_eval.recall_weighted:.6f} F1={final_eval.f1_weighted:.6f}\n"
        )
        f.write(
            f"fire-positive: roc_auc={final_eval.roc_auc_fire:.6f} pr_auc={final_eval.pr_auc_fire:.6f} "
            f"FNR_fire={final_eval.fnr_fire:.6f} FPR_fire={final_eval.fpr_fire:.6f}\n"
        )
        f.write("per-class:\n")
        for cname in class_names:
            f.write(f"  {cname}: {final_eval.per_class.get(cname)}\n")
        f.write("confusion_matrix:\n")
        f.write(str(final_eval.cm) + "\n")
        f.write(f"inference: {ms_per_img:.6f} ms/img, {fps:.6f} FPS\n")


if __name__ == "__main__":
    main()
