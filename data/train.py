import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from flamedataset import FlameDatasetFrom3Dirs
from resnet18 import MultiModalFlameNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

def main():
    # 1. 数据集准备
    root_dir = r"D:\BaiduNetdiskDownload\fire\data\yolo"

    train_dataset = FlameDatasetFrom3Dirs(
        root_dir=root_dir,
        split="train",
        img_size=128,
        series_length=64,
        gaf_method="summation",
        use_augmentation=True,
    )

    val_dataset = FlameDatasetFrom3Dirs(
        root_dir=root_dir,
        split="val",
        img_size=128,
        series_length=64,
        gaf_method="summation",
        use_augmentation=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=4)

    # 2. 模型、损失、优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFlameNet(num_classes=3).to(device)  # 3类：fire / default / smoke

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total params (M): {total_params / 1e6:.3f} M")
    print(f"Trainable params (M): {trainable_params / 1e6:.3f} M")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        # ========= 训练 =========
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, gafs, labels in train_loader:
            imgs = imgs.to(device)
            gafs = gafs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs, gafs)          # (B, num_classes)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ========= 验证 =========
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for imgs, gafs, labels in val_loader:
                imgs = imgs.to(device)
                gafs = gafs.to(device)
                labels = labels.to(device)

                logits = model(imgs, gafs)
                loss = criterion(logits, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                # ⭐ 把当前 batch 的标签和预测收集到 list 里
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_loss /= val_total
        val_acc = val_correct / val_total

        # 如果验证集为空（一般不会），直接跳过避免 warning
        if len(all_labels) == 0:
            print(f"[Epoch {epoch + 1}/{num_epochs}] val set empty, skip metrics")
        else:
            # accuracy：和 val_acc 应该一致，这里可以对一下
            acc = accuracy_score(all_labels, all_preds)

            # macro：对每个类别算 P/R/F1，再取平均（各类权重相同）
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="macro", zero_division=0
            )

            # weighted：按照每一类在验证集中的样本数量加权平均
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="weighted", zero_division=0
            )

            print(
                f"[Epoch {epoch + 1}/{num_epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"prec_macro={precision_macro:.4f} rec_macro={recall_macro:.4f} f1_macro={f1_macro:.4f}"
            )

            # 每类分别的指标（0: fire, 1: default, 2: smoke）
            prec_cls, rec_cls, f1_cls, support = precision_recall_fscore_support(
                all_labels, all_preds, labels=[0, 1, 2], average=None, zero_division=0
            )
            print("  per-class metrics:")
            print(f"    fire   (0): P={prec_cls[0]:.3f}, R={rec_cls[0]:.3f}, F1={f1_cls[0]:.3f}, N={support[0]}")
            print(f"    default(1): P={prec_cls[1]:.3f}, R={rec_cls[1]:.3f}, F1={f1_cls[1]:.3f}, N={support[1]}")
            print(f"    smoke  (2): P={prec_cls[2]:.3f}, R={rec_cls[2]:.3f}, F1={f1_cls[2]:.3f}, N={support[2]}")

            print(classification_report(
                all_labels, all_preds,
                labels=[0, 1, 2],
                target_names=["fire", "default", "smoke"],
                zero_division=0
            ))


if __name__ == "__main__":
    main()
