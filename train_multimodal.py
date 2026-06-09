import os
import shutil
import argparse
import random
import csv
import json
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
from transformers import AutoTokenizer

from model import MultiModalSceneClassifier, set_text_trainable


class SceneMultiModalDataset(Dataset):
    def __init__(self, embed_dir, text_dir, scene_to_tags, tag_to_idx, tokenizer, max_len=512):
        self.embed_dir = embed_dir
        self.text_dir = text_dir
        self.scene_to_tags = scene_to_tags
        self.tag_to_idx = tag_to_idx
        self.scenes = sorted(scene_to_tags.keys())
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_tags = len(tag_to_idx)

        self.scenes = [
            sid for sid in self.scenes
            if os.path.exists(os.path.join(embed_dir, f"{sid}.npy"))
            and os.path.exists(os.path.join(text_dir, f"{sid}.txt"))
        ]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        sid = self.scenes[idx]

        frames = torch.tensor(
            np.load(os.path.join(self.embed_dir, f"{sid}.npy")),
            dtype=torch.float32
        )

        with open(os.path.join(self.text_dir, f"{sid}.txt"), encoding="utf-8") as f:
            text = f.read()

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = torch.zeros(self.num_tags)
        for t in self.scene_to_tags[sid]:
            labels[self.tag_to_idx[t]] = 1.0

        return (
            sid,
            frames,
            enc["input_ids"].squeeze(0),
            enc["attention_mask"].squeeze(0),
            labels,
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        loss = self.alpha * (1 - pt) ** self.gamma * bce

        return loss.mean()


def collate_fn(batch):
    sids, frames, ids, masks, labels = zip(*batch)

    max_len = max(f.shape[0] for f in frames)
    dim = frames[0].shape[1]

    padded = torch.zeros(len(frames), max_len, dim)
    visual_padding_mask = torch.ones(len(frames), max_len, dtype=torch.bool)

    for i, f in enumerate(frames):
        seq_len = f.shape[0]
        padded[i, :seq_len] = f
        visual_padding_mask[i, :seq_len] = False

    return (
        list(sids),
        padded,
        visual_padding_mask,
        torch.stack(ids),
        torch.stack(masks),
        torch.stack(labels),
    )


def split_train_val_test(scene_to_tags, val_fraction=0.05, test_fraction=0.10, seed=42):
    random.seed(seed)
    scene_ids = list(scene_to_tags.keys())
    random.shuffle(scene_ids)

    n_total = len(scene_ids)
    n_test = int(n_total * test_fraction)
    n_val = int(n_total * val_fraction)

    # test slice first so it stays fixed regardless of the val fraction
    test_ids = set(scene_ids[:n_test])
    val_ids = set(scene_ids[n_test:n_test + n_val])
    train_ids = set(scene_ids[n_test + n_val:])

    return (
        {k: scene_to_tags[k] for k in train_ids},
        {k: scene_to_tags[k] for k in val_ids},
        {k: scene_to_tags[k] for k in test_ids},
    )


def load_annotations(path, tag_list):
    scene_to_tags = {}
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}
    idx_to_tag = {i: t for t, i in tag_to_idx.items()}

    with open(path, encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            sid, tags = line.split(":")
            tags = [t.strip() for t in tags.split(",") if t.strip() in tag_to_idx]
            scene_to_tags[sid.strip()] = tags

    return scene_to_tags, tag_to_idx, idx_to_tag


def save_plot(metrics_path):
    epochs = []

    train_loss = []
    val_loss = []

    precision_micro = []
    recall_micro = []
    f1_micro = []
    accuracy = []

    with open(metrics_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))

            precision_micro.append(float(row["val_precision_micro"]))
            recall_micro.append(float(row["val_recall_micro"]))
            f1_micro.append(float(row["val_f1_micro"]))
            accuracy.append(float(row["val_hamming_acc"]))

    out_dir = os.path.dirname(metrics_path)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, f1_micro, label="F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "f1_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, precision_micro, label="Precision")
    plt.plot(epochs, recall_micro, label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "precision_recall_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(epochs, accuracy, label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_curve.png"))
    plt.close()


def tune_thresholds(all_true, all_probs, min_support=5, min_precision=0.6, min_threshold=0.3, alpha=0.7):
    thresholds = []

    for i in range(all_true.shape[1]):

        y_true = all_true[:, i]
        support = y_true.sum()

        if support < min_support:
            thresholds.append(0.5)
            continue

        best_score = 0
        best_t = 0.5

        for t in np.linspace(0.1, 0.8, 30):

            preds = (all_probs[:, i] > t).astype(int)
            precision = precision_score(y_true, preds, zero_division=0)
            if precision < min_precision:
                continue

            f1 = f1_score(y_true, preds, zero_division=0)

            if f1 > best_score:
                best_score = f1
                best_t = t

        best_t = max(best_t, min_threshold)
        best_t = alpha * 0.5 + (1 - alpha) * best_t
        thresholds.append(best_t)

    return np.array(thresholds)


def validate(model, loader, device, criterion, idx_to_tag, save_dir, epoch, thresholds=None):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(save_dir, f"valid_{epoch}.txt")

    all_true, all_probs = [], []
    total_loss = 0.0
    all_sids = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Valid {epoch}", ncols=100)

        for num, (sids, frames, visual_masks, ids, masks, labels) in enumerate(pbar):
            frames = frames.to(device)
            visual_masks = visual_masks.to(device)
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits = model(frames, visual_masks, ids, masks)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()

            all_true.append(labels.cpu().numpy())
            all_probs.append(probs)
            all_sids.extend(sids)

            pbar.set_postfix({"valid_loss": total_loss / (num + 1)})

    all_true = np.vstack(all_true)
    all_probs = np.vstack(all_probs)

    if thresholds is None:
        thresholds = tune_thresholds(all_true, all_probs)
    all_pred = (all_probs > thresholds).astype(int)

    precision_micro = precision_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    recall_micro = recall_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    f1_micro = f1_score(
        all_true, all_pred, average="micro", zero_division=0
    )
    hamming_acc = 1.0 - hamming_loss(all_true, all_pred)

    val_loss = total_loss / len(loader)

    with open(out_path, "w", encoding="utf-8") as f:
        for i, sid in enumerate(all_sids):

            f.write(f"https://www.youtube.com/watch?v={sid}\n")
            f.write("\tpredicted: ")

            hits = [
                (idx_to_tag[j], all_probs[i, j])
                for j in range(all_probs.shape[1])
                if all_probs[i, j] > thresholds[j]
            ]

            if hits:
                for t, p in sorted(hits, key=lambda x: x[1], reverse=True):
                    f.write(f"{t} ({p:.3f}) ")
            else:
                f.write("No label")

            f.write("\n\tannotated: ")
            for j, v in enumerate(all_true[i]):
                if v == 1:
                    f.write(idx_to_tag[j] + " ")

            f.write("\n\n")

    return val_loss, precision_micro, recall_micro, f1_micro, hamming_acc, thresholds


def test(input_file, output_path):
    from test_pipeline import start_test
    
    class Map(dict):
        def __init__(self, *args, **kwargs):
            super(Map, self).__init__(*args, **kwargs)
            for arg in args:
                if isinstance(arg, dict):
                    for k, v in arg.items():
                        self[k] = v
            if kwargs:
                for k, v in kwargs.items():
                    self[k] = v
        def __getattr__(self, attr):
            return self.get(attr)

    test_args = Map({
        "input_file": input_file,
        "output_dir": output_path,
        "model": os.path.join(output_path, "epoch_best.pt"),
        "tags": "data/label_selection.xlsx",
        "segment_duration": 300,
        "silence_threshold": "-30dB",
        "prediction_threshold": 0,
        "silence_min_duration": 0.7,
        "frame_stride": 10,
        "clip_batch_size": 128,
        "min_tag_occurrances": 2,
        "min_tag_score": 0.05
    })

    os.makedirs(os.path.join(output_path, "segments"), exist_ok=True)

    print(f"Starting test on {input_file}...")
    start_test(test_args)


def train(args):
    TAGS = (
        pd.read_excel(args.tags)["English Label"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    TAGS.insert(0, "No label")
    TAGS = sorted(TAGS)

    scene_to_tags, tag_to_idx, idx_to_tag = load_annotations(args.annotations, TAGS)
    train_s2t, val_s2t, test_s2t = split_train_val_test(scene_to_tags)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    train_ds = SceneMultiModalDataset(
        args.frames, args.texts, train_s2t, tag_to_idx, tokenizer
    )
    val_ds = SceneMultiModalDataset(
        args.frames, args.texts, val_s2t, tag_to_idx, tokenizer
    )
    test_ds = SceneMultiModalDataset(
        args.frames, args.texts, test_s2t, tag_to_idx, tokenizer
    )

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    sample_scene = train_ds.scenes[0]
    sample_emb = np.load(os.path.join(args.frames, f"{sample_scene}.npy"))
    clip_dim = sample_emb.shape[1]

    print(f"CLIP shape: {sample_emb.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalSceneClassifier(clip_dim, len(tag_to_idx), fusion=args.fusion).to(device)

    set_text_trainable(model.text_encoder, args.unfreeze_text_layers)

    text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and not n.startswith("text_encoder.")
    ]
    print(f"Trainable text-encoder layers: {args.unfreeze_text_layers} "
          f"({sum(p.numel() for p in text_params):,} params)")

    param_groups = [{"params": other_params, "lr": args.lr}]
    if text_params:
        param_groups.append({"params": text_params, "lr": args.text_lr})

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(param_groups)

    os.makedirs(args.save_dir, exist_ok=True)
    for script_name in glob("*.py"):
        shutil.copy(script_name, args.save_dir)
        
    metrics_path = os.path.join(args.save_dir, "metrics.csv")

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_precision_micro",
            "val_recall_micro",
            "val_f1_micro",
            "val_hamming_acc"
        ])

    best = -1.0
    early_stop_count = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch}", ncols=100)
        for num, (_, frames, visual_masks, ids, masks, labels) in enumerate(pbar):
            frames = frames.to(device)
            visual_masks = visual_masks.to(device)
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(frames, visual_masks, ids, masks)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

            pbar.set_postfix({"train_loss": loss_sum / (num + 1), "ES counter": early_stop_count})

        val_loss, precision_micro, recall_micro, f1_micro, hamming_acc, thresholds = validate(
            model, val_loader, device, criterion, idx_to_tag, args.save_dir, epoch
        )

        with open(metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                loss_sum / len(train_loader),
                val_loss,
                precision_micro,
                recall_micro,
                f1_micro,
                hamming_acc
            ])

        torch.save(model.state_dict(), os.path.join(args.save_dir, "epoch_latest.pt"))
        save_plot(metrics_path)

        if f1_micro > best:
            best = f1_micro

            torch.save(model.state_dict(), os.path.join(args.save_dir, "epoch_best.pt"))
            np.save(os.path.join(args.save_dir, "best_thresholds.npy"), thresholds)

            if args.early_stop != 0:
                early_stop_count = 0

        elif args.early_stop != 0:
            early_stop_count += 1
            if early_stop_count == args.early_stop:
                break

        time.sleep(1)

    final_test(model, test_loader, device, criterion, idx_to_tag, args.save_dir)


def final_test(model, test_loader, device, criterion, idx_to_tag, save_dir):
    best_path = os.path.join(save_dir, "epoch_best.pt")
    print(f"\nLoading best model from {best_path} for final test...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    # use the thresholds tuned on validation (re-tuning on test would leak)
    thresholds = np.load(os.path.join(save_dir, "best_thresholds.npy"))

    test_loss, precision_micro, recall_micro, f1_micro, hamming_acc, _ = validate(
        model, test_loader, device, criterion, idx_to_tag, save_dir, "test", thresholds=thresholds
    )

    final_metrics = {
        "test_loss": test_loss,
        "test_precision_micro": precision_micro,
        "test_recall_micro": recall_micro,
        "test_f1_micro": f1_micro,
        "test_hamming_acc": hamming_acc,
    }

    with open(os.path.join(save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("Final test metrics (best model, val-tuned thresholds):")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="./extracted_data/embeddings")
    parser.add_argument("--texts", default="./extracted_data/texts")
    parser.add_argument("--annotations", default="./data/annotations.txt")
    parser.add_argument("--tags", default="./data/label_selection.xlsx")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fusion", default="film", choices=["cross_attn", "concat", "add", "film"])
    parser.add_argument("--unfreeze-text-layers", type=int, default=0,
                        help="Number of top DistilBERT transformer blocks to fine-tune (0 = frozen)")
    parser.add_argument("--text-lr", type=float, default=2e-5,
                        help="Learning rate for the unfrozen text-encoder layers")
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--test", action="store_true", help="Run in test mode")

    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    if not args.test:
        train(args)

    if args.test_file is not None:
        test(args.test_file, args.save_dir)
