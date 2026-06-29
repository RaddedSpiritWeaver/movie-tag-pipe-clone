import torch
import subprocess
import json
import re
import os
import clip
import argparse
import numpy as np
import cv2 as cv
import pandas as pd
import librosa
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from transformers import AutoTokenizer, ClapModel, ClapProcessor

from model import (
    MultiModalSceneClassifier, TextSceneClassifier, VisionSceneClassifier,
    AudioSceneClassifier, DEFAULT_TEXT_MODEL,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

CLAP_MODEL = "laion/clap-htsat-unfused"
AUDIO_CHUNK_SECONDS = 10
AUDIO_BATCH_CHUNKS = 16

class SceneInferenceDataset(Dataset):
    def __init__(self, embed_dir, text_dir, tokenizer, max_len=512, audio_dir=None):
        self.embed_dir = Path(embed_dir)
        self.text_dir = Path(text_dir)
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.ids = [
            p.stem
            for p in self.embed_dir.glob("segment_*.npy")
            if not p.stem.endswith(".audio")
            and (self.text_dir / f"{p.stem}.txt").exists()
            and (self.audio_dir is None or (self.audio_dir / f"{p.stem}.audio.npy").exists())
        ]

        self.ids.sort()
        assert len(self.ids) > 0, f"No matching npy/txt pairs found in {embed_dir}"

        first_emb = np.load(self.embed_dir / f"{self.ids[0]}.npy")
        self.clip_dim = first_emb.shape[1]

        self.audio_dim = None
        if self.audio_dir is not None:
            self.audio_dim = np.load(self.audio_dir / f"{self.ids[0]}.audio.npy").shape[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        frames = torch.tensor(np.load(self.embed_dir / f"{sid}.npy"), dtype=torch.float32)

        with open(self.text_dir / f"{sid}.txt", encoding="utf-8") as f:
            text = f.read()

        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )

        if self.audio_dir is None:
            return sid, frames, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

        audio = torch.tensor(np.load(self.audio_dir / f"{sid}.audio.npy"), dtype=torch.float32)
        return sid, frames, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0), audio


def _pad_sequence(seqs):
    max_len = max(s.shape[0] for s in seqs)
    dim = seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_len, dim)
    padding_mask = torch.ones(len(seqs), max_len, dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :s.shape[0]] = s
        padding_mask[i, :s.shape[0]] = False
    return padded, padding_mask


def collate_fn(batch):
    sids, frames, ids, masks = zip(*batch)
    padded, visual_padding_mask = _pad_sequence(frames)
    return list(sids), padded, visual_padding_mask, torch.stack(ids), torch.stack(masks)


def multimodal_collate_fn(batch):
    sids, frames, ids, masks, audio = zip(*batch)
    padded, visual_padding_mask = _pad_sequence(frames)
    audio_padded, audio_padding_mask = _pad_sequence(audio)
    return (list(sids), padded, visual_padding_mask, torch.stack(ids),
            torch.stack(masks), audio_padded, audio_padding_mask)


class SceneAudioInferenceDataset(Dataset):
    def __init__(self, audio_dir):
        self.audio_dir = Path(audio_dir)
        self.ids = sorted(
            p.stem[:-len(".audio")]
            for p in self.audio_dir.glob("segment_*.audio.npy")
        )
        assert len(self.ids) > 0, f"No .audio.npy segments found in {audio_dir}"
        self.audio_dim = np.load(self.audio_dir / f"{self.ids[0]}.audio.npy").shape[1]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        audio = torch.tensor(np.load(self.audio_dir / f"{sid}.audio.npy"), dtype=torch.float32)
        return sid, audio


def audio_collate_fn(batch):
    sids, audio = zip(*batch)
    audio_padded, audio_padding_mask = _pad_sequence(audio)
    return list(sids), audio_padded, audio_padding_mask


class SceneTextInferenceDataset(Dataset):
    def __init__(self, text_dir, tokenizer, max_len=512):
        self.text_dir = Path(text_dir)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.ids = sorted(p.stem for p in self.text_dir.glob("segment_*.txt"))
        assert len(self.ids) > 0, f"No .txt segments found in {text_dir}"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        with open(self.text_dir / f"{sid}.txt", encoding="utf-8") as f:
            text = f.read()

        enc = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return sid, enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


def text_collate_fn(batch):
    sids, ids, masks = zip(*batch)
    return list(sids), torch.stack(ids), torch.stack(masks)


def run(cmd):
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    return result.stdout, result.stderr


def get_video_duration(file):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", file]
    out, _ = run(cmd)
    return float(json.loads(out)["format"]["duration"])


def detect_silences(file, threshold="-30dB", min_duration=0.7):
    cmd = ["ffmpeg", "-i", file, "-af", f"silencedetect=noise={threshold}:d={min_duration}", "-f", "null", "-"]
    _, err = run(cmd)
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start: (\d+\.?\d*)", err)]
    return starts


def get_best_streams(input_file, debug=True):
    cmd = [
        "ffprobe", "-v", "error",
        "-show_streams",
        "-of", "json",
        input_file
    ]
    out, _ = run(cmd)
    data = json.loads(out)

    best_audio = None
    best_audio_channels = -1

    eng_sub = None
    eng_sdh_sub = None

    if debug:
        print("\n=== STREAM INSPECTION ===")

    for s in data["streams"]:
        idx = s["index"]
        codec_type = s.get("codec_type")
        tags = s.get("tags", {})
        lang = tags.get("language", "unknown")
        title = tags.get("title", "")
        channels = s.get("channels", "-")

        if debug:
            print(
                f"[{idx}] {codec_type.upper()} | "
                f"lang={lang} | "
                f"channels={channels} | "
                f"title='{title}'"
            )

        if codec_type == "audio" and lang == "eng":
            if channels > best_audio_channels:
                best_audio = idx
                best_audio_channels = channels

        if codec_type == "subtitle" and lang == "eng":
            if eng_sub is None:
                eng_sub = idx

            if "sdh" in title.lower() or "hearing" in title.lower():
                eng_sdh_sub = idx

    best_sub = eng_sdh_sub if eng_sdh_sub is not None else eng_sub

    if debug:
        print("\n=== SELECTED STREAMS ===")
        print(f"Audio stream: {best_audio}")
        print(f"Subtitle stream: {best_sub}")
        print("========================\n")

    return best_audio, best_sub


def split_video(input_file, segments, output_dir):
    Path(output_dir).mkdir(exist_ok=True)

    audio_idx, sub_idx = get_best_streams(input_file)

    if audio_idx is None:
        print("WARNING: No English audio found")
    if sub_idx is None:
        print("WARNING: No English subtitles found")

    for i, (start, end) in enumerate(tqdm(segments, desc="Splitting Video")):
        duration = end - start
        output_video = f"{output_dir}/segment_{i:03d}.mkv"
        output_sub = f"{output_dir}/segment_{i:03d}.srt"

        if not os.path.exists(output_video):
            cmd = [
                "ffmpeg", "-ss", str(start), "-i", input_file,
                "-t", str(duration),
                "-map", "0:v:0"
            ]

            if audio_idx is not None:
                cmd += ["-map", f"0:{audio_idx}"]

            cmd += ["-c", "copy", output_video, "-y"]

            run(cmd)

        if sub_idx is not None and not os.path.exists(output_sub):
            run([
                "ffmpeg", "-ss", str(start), "-i", input_file,
                "-t", str(duration),
                "-map", f"0:{sub_idx}",
                output_sub, "-y"
            ])

        if os.path.exists(output_sub):
            with open(output_sub, "r", encoding="utf-8") as f:
                content = f.read()

            content = re.sub(r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> .*', '', content)
            content = re.sub(r'<[^>]*>', '', content)

            clean_text = " ".join([
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().isdigit()
            ])

            with open(output_sub.replace(".srt", ".txt"), "w", encoding="utf-8") as f:
                f.write(clean_text)


def segment_video(args):
    duration = get_video_duration(args.input_file)
    silence_starts = detect_silences(
        args.input_file, 
        threshold=args.silence_threshold, 
        min_duration=args.silence_min_duration
    )
    segments = []
    current_start = 0
    while current_start < duration:
        target = current_start + args.segment_duration
        if target >= duration:
            segments.append((current_start, duration))
            break
        cut_point = min(silence_starts, key=lambda x: abs(x - target)) if silence_starts else target
        segments.append((current_start, cut_point))
        current_start = cut_point
    split_video(args.input_file, segments, args.output_dir)


@torch.inference_mode()
def embed_segments(args):
    clip_model, clip_preprocess = clip.load("ViT-L/14", device=device)
    segments = sorted(glob(os.path.join(args.output_dir, "segment_*.mkv")))
    
    for video in tqdm(segments, desc="Embedding"):
        output_path = video.replace(".mkv", ".npy")
        if os.path.exists(output_path): continue

        cap = cv.VideoCapture(video)
        all_embs, batch, frame_idx = [], [], 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % args.frame_stride == 0:
                frame = cv.cvtColor(cv.resize(frame, (224, 224)), cv.COLOR_BGR2RGB)
                img = torch.from_numpy(frame).permute(2,0,1).float() / 255.0
                batch.append(clip_preprocess.transforms[-1](img))
            
            if len(batch) == args.clip_batch_size:
                with torch.amp.autocast("cuda"):
                    e = clip_model.encode_image(torch.stack(batch).to(device))
                all_embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
                batch.clear()
            frame_idx += 1
        
        if batch:
            with torch.amp.autocast("cuda"):
                e = clip_model.encode_image(torch.stack(batch).to(device))
            all_embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        
        cap.release()
        if all_embs:
            np.save(output_path, torch.cat(all_embs).numpy())


@torch.inference_mode()
def embed_audio_segments(args):
    """CLAP audio-embedding sequence per video segment -> segment_*.audio.npy."""
    clap = ClapModel.from_pretrained(CLAP_MODEL).to(device).eval()
    processor = ClapProcessor.from_pretrained(CLAP_MODEL)
    sr = processor.feature_extractor.sampling_rate
    chunk_len = AUDIO_CHUNK_SECONDS * sr
    min_len = sr

    segments = sorted(glob(os.path.join(args.output_dir, "segment_*.mkv")))
    for video in tqdm(segments, desc="Audio embedding"):
        output_path = video.replace(".mkv", ".audio.npy")
        if os.path.exists(output_path):
            continue

        # extract the segment audio to a temp wav, then load at CLAP's rate
        wav_path = video.replace(".mkv", ".clap.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video,
             "-vn", "-ac", "1", "-ar", str(sr), str(wav_path)],
            check=False,
        )
        if not os.path.exists(wav_path):
            np.save(output_path, np.zeros((1, clap.config.projection_dim), dtype="float32"))
            continue
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        os.remove(wav_path)
        if wav.size < min_len:
            wav = np.zeros(min_len, dtype=np.float32)

        chunks = [wav[s:s + chunk_len] for s in range(0, len(wav), chunk_len)]
        chunks = [c for c in chunks if len(c) >= min_len] or [wav]

        embs = []
        for i in range(0, len(chunks), AUDIO_BATCH_CHUNKS):
            group = chunks[i:i + AUDIO_BATCH_CHUNKS]
            inp = processor(audio=group, sampling_rate=sr, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            pooled = clap.audio_model(**inp).pooler_output
            e = clap.audio_projection(pooled)
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu())

        np.save(output_path, torch.cat(embs).numpy().astype("float32"))


@torch.no_grad()
def test(args):
    tags_df = pd.read_excel(args.tags)["English Label"].dropna().astype(str).unique().tolist()
    TAGS = sorted(["No label"] + tags_df)
    idx_to_tag = {i: t for i, t in enumerate(TAGS)}

    text_model = getattr(args, "text_model", DEFAULT_TEXT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    model_type = getattr(args, "model_type", "multimodal")

    if model_type == "text":
        dataset = SceneTextInferenceDataset(args.output_dir, tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=text_collate_fn)
        model = TextSceneClassifier(len(TAGS), text_model=text_model).to(device)
    elif model_type == "audio":
        dataset = SceneAudioInferenceDataset(args.output_dir)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=audio_collate_fn)
        model = AudioSceneClassifier(dataset.audio_dim, len(TAGS)).to(device)
    elif model_type == "frame":
        dataset = SceneInferenceDataset(args.output_dir, args.output_dir, tokenizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
        model = VisionSceneClassifier(dataset.clip_dim, len(TAGS)).to(device)
    else:
        dataset = SceneInferenceDataset(args.output_dir, args.output_dir, tokenizer, audio_dir=args.output_dir)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=multimodal_collate_fn)
        model = MultiModalSceneClassifier(dataset.clip_dim, len(TAGS), dataset.audio_dim, text_model=text_model).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    if float(args.prediction_threshold) == 0:
        model_dir = os.path.dirname(args.model)
        thresholds_path = os.path.join(model_dir, "best_thresholds.npy")

        if not os.path.exists(thresholds_path):
            raise FileNotFoundError(
                f"Auto threshold requested but not found: {thresholds_path}"
            )

        thresholds = np.load(thresholds_path)

        if len(thresholds) != len(TAGS):
            raise ValueError(
                f"Threshold size mismatch: {len(thresholds)} vs {len(TAGS)} tags"
            )

        print("Using tuned per-tag thresholds.")
    else:
        thresholds = np.full(len(TAGS), float(args.prediction_threshold))
        print(f"Using fixed threshold = {args.prediction_threshold}")

    tag_counts = defaultdict(int)
    tag_prob_sums = defaultdict(float)
    total_segments = 0

    results_file = os.path.join(args.output_dir, f"{args.movie_name}_results.txt")
    with open(results_file, "w", encoding="utf-8") as f:
        for batch in tqdm(loader, desc="Classifying"):
            if model_type == "text":
                sids, ids, masks = batch
                logits = model(ids.to(device), masks.to(device))
            elif model_type == "audio":
                sids, audio, audio_masks = batch
                logits = model(audio.to(device), audio_masks.to(device))
            elif model_type == "frame":
                sids, frames, visual_masks, ids, masks = batch
                logits = model(frames.to(device), visual_masks.to(device))
            else:
                sids, frames, visual_masks, ids, masks, audio, audio_masks = batch
                logits = model(frames.to(device), visual_masks.to(device), ids.to(device),
                               masks.to(device), audio.to(device), audio_masks.to(device))
            probs = torch.sigmoid(logits)[0].cpu().numpy()

            preds = sorted(
                [
                    (idx_to_tag[i], p)
                    for i, p in enumerate(probs)
                    if p >= thresholds[i]
                ],
                key=lambda x: x[1],
                reverse=True
            )

            sid = sids[0]
            total_segments += 1

            labels_str = ", ".join(t for t, _ in preds) if preds else "No label"
            f.write(f"{sid}.mkv: {labels_str}\n")

            for tag, prob in preds:
                tag_counts[tag] += 1
                tag_prob_sums[tag] += prob

        f.write("\n" + "=" * 60 + "\n")
        f.write("DOMINANT TAGS\n")
        f.write("=" * 60 + "\n")

        dominant_tags = []

        for tag in tag_counts:
            count = tag_counts[tag]
            avg_prob = tag_prob_sums[tag] / count
            freq_ratio = count / total_segments
            dominance_score = freq_ratio * avg_prob

            dominant_tags.append(
                (tag, count, freq_ratio, avg_prob, dominance_score)
            )

        dominant_tags = [
            t for t in dominant_tags
            if t[1] >= int(args.min_tag_occurrances)
            and t[4] >= float(args.min_tag_score)
        ]

        dominant_tags.sort(key=lambda x: x[4], reverse=True)

        if dominant_tags:
            for tag, count, freq_ratio, avg_prob, score in dominant_tags:
                f.write(
                    f"{tag} | "
                    f"segments: {count}/{total_segments} | "
                    f"freq: {freq_ratio:.2f} | "
                    f"avg_conf: {avg_prob:.2f} | "
                    f"score: {score:.3f}\n"
                )
        else:
            f.write("No dominant tags detected.\n")


def start_test(args):
    args.movie_name = Path(args.input_file).stem
    args.output_dir = os.path.join(args.output_dir, "segments", args.movie_name)
    
    print(f"Processing Movie: {args.movie_name}")
    print(f"Output Directory: {args.output_dir}")

    model_type = getattr(args, "model_type", "multimodal")

    segment_video(args)
    if model_type != "audio":
        embed_segments(args)
    if model_type in ("audio", "multimodal"):
        embed_audio_segments(args)
    test(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL,
                        help="Text encoder used at train time (must match the checkpoint)")
    parser.add_argument("--model-type", default="multimodal",
                        choices=["frame", "text", "audio", "multimodal"],
                        help="Must match how the checkpoint was trained")
    parser.add_argument("--tags", default="data/label_selection.xlsx")
    parser.add_argument("--segment-duration", default=300)
    parser.add_argument("--silence-threshold", default="-30dB")
    parser.add_argument("--prediction-threshold", default=0.5, help="0 for auto")
    parser.add_argument("--silence-min-duration", default=0.7)
    parser.add_argument("--frame_stride", default=10)
    parser.add_argument("--clip-batch-size", default=128)
    parser.add_argument("--min-tag-occurrances", default=2)
    parser.add_argument("--min-tag-score", default=0.05)

    args = parser.parse_args()
    start_test(args)