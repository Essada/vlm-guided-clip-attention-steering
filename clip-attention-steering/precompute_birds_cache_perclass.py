import json
import os
import sys
import tempfile

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from ConformationCLIP_ViT import CLIPViT
from test_birds import (
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    CUBTestDataset,
    bbox_to_patch_indices,
)
from test_parts_steering import load_grounding_dino

DATA_ROOT = os.environ.get(
    "CUB_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "CUB_200_2011"),
)
QUERIES_PATH = os.path.join(os.path.dirname(__file__), "cub_class_queries.json")
MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-16")
N_SAMPLES = int(os.environ.get("CUB_PRECOMPUTE_SAMPLES", "3000"))
BATCH_SIZE = 8
QUERY_TAG = "perclass-tight"
CACHE_DIR = DATA_ROOT
CACHE_VERSION = 1
DINO_BOX_THRESHOLD = 0.35
DINO_TEXT_THRESHOLD = 0.25
TOP_K_BOXES = 3
PATCH_OVERLAP_FRAC = 0.90
CHECKPOINT_EVERY = int(os.environ.get("CUB_PRECOMPUTE_CHECKPOINT_EVERY", "500"))


def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def collate_with_pil_and_path(batch):
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    pils = [b[2] for b in batch]
    paths = [b[3] for b in batch]
    return images, labels, pils, paths


class CUBTestDatasetWithPath(CUBTestDataset):
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = Image.open(path).convert("RGB")
        tensor = self.transform(pil) if self.transform else pil
        return tensor, label, pil, path


def sanitize_model_name(name: str) -> str:
    return name.lower().replace("/", "-").replace(" ", "-")


def cache_path_for(model_name, n_samples):
    return os.path.join(
        CACHE_DIR,
        f"cub_precompute_{sanitize_model_name(model_name)}_{n_samples}_{QUERY_TAG}.pt",
    )


def checkpoint_path_for(model_name, n_samples):
    return os.path.join(
        CACHE_DIR,
        f"cub_precompute_{sanitize_model_name(model_name)}_{n_samples}_{QUERY_TAG}.partial.pt",
    )


def atomic_torch_save(payload, out_path):
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(out_path)}.",
                                suffix=".tmp", dir=out_dir)
    os.close(fd)
    try:
        torch.save(payload, tmp, _use_new_zipfile_serialization=False)
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def detect_boxes_and_tokens(pil_img, query, gd_proc, gd_model, device,
                              patch_size, grid_size):
    orig_w, orig_h = pil_img.size
    inputs = gd_proc(images=pil_img, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)
    try:
        results = gd_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=DINO_BOX_THRESHOLD, text_threshold=DINO_TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]
    except TypeError:
        results = gd_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=DINO_BOX_THRESHOLD, text_threshold=DINO_TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]
    boxes = results["boxes"].cpu()
    scores = results["scores"].cpu() if "scores" in results else None
    if boxes.shape[0] == 0:
        return list(range(grid_size * grid_size)), False
    if scores is not None and boxes.shape[0] > TOP_K_BOXES:
        topk = scores.argsort(descending=True)[:TOP_K_BOXES]
        boxes = boxes[topk]
    selected = set()
    for box in boxes.tolist():
        selected.update(bbox_to_patch_indices(
            box, orig_w, orig_h, patch_size=patch_size, grid_size=grid_size,
            overlap_frac=PATCH_OVERLAP_FRAC,
        ))
    return (list(selected), True) if selected else (list(range(grid_size * grid_size)), False)


def save_payload(out_path, model_name, root, n_samples, clip,
                  class_texts, class_text_feats,
                  imgs_all, labels_all, paths_all,
                  tokens_all, detected_all, queries_used,
                  complete):
    payload = {
        "meta": {
            "cache_version": CACHE_VERSION,
            "model_name": model_name,
            "data_root": os.path.abspath(root),
            "n_samples": n_samples,
            "query": "per-class (see queries_used)",
            "patch_size": clip.patch_size,
            "grid_size": clip.grid_size,
            "num_spatial_tokens": clip.num_spatial_tokens,
            "box_threshold": DINO_BOX_THRESHOLD,
            "text_threshold": DINO_TEXT_THRESHOLD,
            "top_k_boxes": TOP_K_BOXES,
            "patch_overlap_frac": PATCH_OVERLAP_FRAC,
            "complete": complete,
            "num_cached": len(paths_all),
        },
        "data": {
            "imgs": torch.cat(imgs_all) if imgs_all else torch.empty((0, 3, 224, 224)),
            "labels": torch.cat(labels_all) if labels_all else torch.empty((0,), dtype=torch.long),
            "image_paths": paths_all,
            "tokens": tokens_all,
            "dino_detected": detected_all,
            "class_texts": class_texts,
            "class_text_feats": class_text_feats,
            "queries_used": queries_used,
        },
    }
    atomic_torch_save(payload, out_path)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    device = select_device()

    if not os.path.exists(QUERIES_PATH):
        sys.exit(f"Missing {QUERIES_PATH}. Run generate_class_queries.py first.")
    with open(QUERIES_PATH) as f:
        class_queries = json.load(f)
    print(f"Loaded {len(class_queries)} per-class queries from {QUERIES_PATH}")

    clip = CLIPViT(model_name=MODEL_NAME).to(device)
    clip.eval()

    dataset = CUBTestDatasetWithPath(root, transform=clip.preprocess)
    total_test = len(dataset)
    n_samples = min(N_SAMPLES, total_test)

    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_test, generator=rng)[:n_samples].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_with_pil_and_path,
    )

    gd_proc, gd_model = load_grounding_dino(device)

    class_texts = [
        f"a photo of a {c.replace('_', ' ').split('.', 1)[-1].strip()}"
        for c in dataset.classes
    ]
    class_text_feats = clip.encode_text(class_texts).cpu()

    folder_for_label = dataset.classes

    missing = [c for c in dataset.classes if c not in class_queries]
    if missing:
        sys.exit(
            f"Missing per-class queries for {len(missing)} classes. "
            f"Examples: {missing[:5]}\n"
            f"Re-run generate_class_queries.py to fill them in."
        )

    imgs_all, labels_all, paths_all = [], [], []
    tokens_all, detected_all, queries_used = [], [], []
    out_path = cache_path_for(MODEL_NAME, n_samples)
    partial_path = checkpoint_path_for(MODEL_NAME, n_samples)

    for images, labels, pils, paths in tqdm(loader, desc="Precompute CUB cache (perclass)"):
        imgs_all.append(images.cpu())
        labels_all.append(labels.cpu())
        paths_all.extend(paths)

        for pil, lbl in zip(pils, labels.tolist()):
            class_folder = folder_for_label[lbl]
            query = class_queries[class_folder]
            tokens, detected = detect_boxes_and_tokens(
                pil, query, gd_proc, gd_model, device,
                patch_size=clip.patch_size, grid_size=clip.grid_size,
            )
            tokens_all.append(tokens)
            detected_all.append(detected)
            queries_used.append(query)

        if CHECKPOINT_EVERY > 0 and len(paths_all) % CHECKPOINT_EVERY == 0:
            save_payload(
                partial_path, MODEL_NAME, root, n_samples, clip,
                class_texts, class_text_feats,
                imgs_all, labels_all, paths_all,
                tokens_all, detected_all, queries_used,
                complete=False,
            )
            print(f"Checkpoint saved -> {partial_path}  ({len(paths_all)} images)")

    save_payload(
        out_path, MODEL_NAME, root, n_samples, clip,
        class_texts, class_text_feats,
        imgs_all, labels_all, paths_all,
        tokens_all, detected_all, queries_used,
        complete=True,
    )

    print(f"\nSaved -> {out_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Test images cached: {n_samples} / {total_test}")
    print(f"DINO detected boxes on {sum(detected_all)}/{len(detected_all)} images")


if __name__ == "__main__":
    main()
