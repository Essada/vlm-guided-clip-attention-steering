import os
import sys
import tempfile

import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ConformationCLIP_ViT import CLIPViT
from test_birds import bbox_to_patch_indices
from test_parts_steering import load_grounding_dino
from test_tapc_pasta import (
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    PRECOMPUTE_CACHE_VERSION,
    YesNoVQADataset,
    collate_fn,
    extract_dino_query,
    get_precompute_cache_path,
)
from test_tapc import generate_yes_no_statements

DATA_ROOT = os.environ.get("VQA_DATA_ROOT", os.path.expanduser("~/data/vqa"))
MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-16")
N_SAMPLES = int(os.environ.get("TAPC_PRECOMPUTE_SAMPLES", "5000"))
BATCH_SIZE = 8
CHECKPOINT_EVERY = int(os.environ.get("TAPC_PRECOMPUTE_CHECKPOINT_EVERY", "1000"))


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def checkpoint_path_for(cache_path: str) -> str:
    base, ext = os.path.splitext(cache_path)
    return f"{base}.partial{ext}"


def atomic_torch_save(payload, out_path: str):
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(out_path)}.",
        suffix=".tmp",
        dir=out_dir,
    )
    os.close(fd)
    try:
        torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def detect_boxes_and_tokens(pil_img, gd_proc, gd_model, device, query, patch_size, grid_size):
    orig_w, orig_h = pil_img.size
    inputs = gd_proc(images=pil_img, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gd_model(**inputs)

    try:
        results = gd_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]
    except TypeError:
        results = gd_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]

    boxes = results["boxes"].cpu()
    if boxes.shape[0] == 0:
        return list(range(grid_size * grid_size)), False

    selected = set()
    for box in boxes.tolist():
        selected.update(
            bbox_to_patch_indices(
                box,
                orig_w,
                orig_h,
                patch_size=patch_size,
                grid_size=grid_size,
            )
        )
    return (list(selected), True) if selected else (list(range(grid_size * grid_size)), False)


def save_payload(
    out_path,
    model_name,
    root,
    n_samples,
    clip,
    imgs_all,
    tokens_all,
    detected_all,
    yes_feats_all,
    no_feats_all,
    gts_all,
    questions_all,
    keys_all,
    paths_all,
    complete,
):
    payload = {
        "meta": {
            "cache_version": PRECOMPUTE_CACHE_VERSION,
            "model_name": model_name,
            "pool_size": n_samples,
            "patch_size": clip.patch_size,
            "grid_size": clip.grid_size,
            "num_spatial_tokens": clip.num_spatial_tokens,
            "data_root": os.path.abspath(root),
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
            "complete": complete,
            "num_cached": len(keys_all),
        },
        "data": {
            "imgs": torch.cat(imgs_all) if imgs_all else torch.empty((0, 3, 224, 224)),
            "tokens": tokens_all,
            "dino_detected": detected_all,
            "yes_feats": torch.stack(yes_feats_all) if yes_feats_all else torch.empty((0, 512)),
            "no_feats": torch.stack(no_feats_all) if no_feats_all else torch.empty((0, 512)),
            "gts": gts_all,
            "questions": questions_all,
            "keys": keys_all,
            "image_paths": paths_all,
        },
    }
    atomic_torch_save(payload, out_path)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    device = select_device()

    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        print("ERROR: Ollama is not running. Start with: ollama serve")
        sys.exit(1)

    clip = CLIPViT(model_name=MODEL_NAME).to(device)
    clip.eval()

    full_ds = YesNoVQADataset(root, clip.preprocess)
    total_available = len(full_ds)
    n_samples = min(N_SAMPLES, total_available)

    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_available, generator=rng)[:n_samples].tolist()
    subset = torch.utils.data.Subset(full_ds, indices)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    gd_proc, gd_model = load_grounding_dino(device)

    out_path = get_precompute_cache_path(MODEL_NAME, n_samples)
    partial_path = checkpoint_path_for(out_path)

    imgs_all = []
    tokens_all = []
    detected_all = []
    yes_feats_all = []
    no_feats_all = []
    gts_all = []
    questions_all = []
    keys_all = []
    paths_all = []

    for imgs, gts, qs, pils, keys, paths in tqdm(loader, desc="Precompute TAP-C cache"):
        imgs_all.append(imgs.cpu())
        gts_all.extend(gts)
        questions_all.extend(qs)
        keys_all.extend(keys)
        paths_all.extend(paths)

        for q, pil in zip(qs, pils):
            dino_query = extract_dino_query(q)
            yes_stmt, no_stmt = generate_yes_no_statements(q)

            tokens, detected = detect_boxes_and_tokens(
                pil, gd_proc, gd_model, device, dino_query,
                patch_size=clip.patch_size, grid_size=clip.grid_size,
            )
            tokens_all.append(tokens)
            detected_all.append(detected)

            with torch.no_grad():
                tf = clip.encode_text([yes_stmt, no_stmt])
            yes_feats_all.append(tf[0].cpu())
            no_feats_all.append(tf[1].cpu())

        if CHECKPOINT_EVERY > 0 and len(keys_all) % CHECKPOINT_EVERY == 0:
            save_payload(
                partial_path,
                MODEL_NAME,
                root,
                n_samples,
                clip,
                imgs_all,
                tokens_all,
                detected_all,
                yes_feats_all,
                no_feats_all,
                gts_all,
                questions_all,
                keys_all,
                paths_all,
                complete=False,
            )
            print(f"Checkpoint saved -> {partial_path}  ({len(keys_all)} samples)")

    save_payload(
        partial_path,
        MODEL_NAME,
        root,
        n_samples,
        clip,
        imgs_all,
        tokens_all,
        detected_all,
        yes_feats_all,
        no_feats_all,
        gts_all,
        questions_all,
        keys_all,
        paths_all,
        complete=False,
    )
    print(f"Final checkpoint saved -> {partial_path}  ({len(keys_all)} samples)")

    save_payload(
        out_path,
        MODEL_NAME,
        root,
        n_samples,
        clip,
        imgs_all,
        tokens_all,
        detected_all,
        yes_feats_all,
        no_feats_all,
        gts_all,
        questions_all,
        keys_all,
        paths_all,
        complete=True,
    )

    print(f"Saved -> {out_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device}")
    print(f"Yes/No samples cached: {n_samples} / {total_available}")
    print(f"Box threshold: {BOX_THRESHOLD}  Text threshold: {TEXT_THRESHOLD}")


if __name__ == "__main__":
    main()
