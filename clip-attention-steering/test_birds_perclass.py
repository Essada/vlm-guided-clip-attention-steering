import os
import sys

import torch
from torch.utils.data import DataLoader

from ConformationCLIP_ViT import CLIPViT
from test_birds import (
    HARDCODED_TOP_HEADS,
    DATA_ROOT,
    BATCH_SIZE,
    CachedBirdSubset,
    collate_cached,
    sanitize_model_name,
)
from test_parts_steering import eval_plain, eval_pasta_cached, profile_heads

MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B-16")
CACHE_SAMPLES = 3000
N_PROFILE = 500
N_EVAL = 2500
PASTA_ALPHAS = [0.3, 0.1, 0.05]
HEAD_COUNTS = [5, 10]
PROFILE_LAYERS = list(range(4, 12))
PROFILE_ALPHA = 0.1
QUERY_TAG = "perclass"
CACHE_VERSION = 1


def cache_path_for(root, model_name, n_samples):
    return os.path.join(
        root,
        f"cub_precompute_{sanitize_model_name(model_name)}_{n_samples}_{QUERY_TAG}.pt",
    )


def load_cache(root, model_name, expected_samples):
    path = cache_path_for(root, model_name, expected_samples)
    if not os.path.exists(path):
        return None, path, f"missing file: {path}"
    payload = torch.load(path, map_location="cpu")
    meta = payload.get("meta", {})
    if not meta.get("complete", False):
        return None, path, "cache not marked complete"
    if meta.get("n_samples") != expected_samples:
        return None, path, f"n_samples mismatch (got {meta.get('n_samples')})"
    if int(payload["data"]["imgs"].shape[0]) < N_PROFILE + N_EVAL:
        return None, path, f"only {payload['data']['imgs'].shape[0]} cached, need {N_PROFILE + N_EVAL}"
    return payload, path, None


def stratified_profile_split(labels_tensor, per_class, seed=42):
    g = torch.Generator().manual_seed(seed)
    labels = labels_tensor.tolist()
    by_class = {}
    for i, y in enumerate(labels):
        by_class.setdefault(y, []).append(i)

    profile_idx, eval_pool = [], []
    for y, idxs in by_class.items():
        perm = torch.randperm(len(idxs), generator=g).tolist()
        shuffled = [idxs[j] for j in perm]
        profile_idx.extend(shuffled[:per_class])
        eval_pool.extend(shuffled[per_class:])

    eval_perm = torch.randperm(len(eval_pool), generator=g).tolist()
    return profile_idx, [eval_pool[j] for j in eval_perm]


def build_loaders(payload):
    data = payload["data"]
    n = int(data["imgs"].shape[0])
    labels_all = data["labels"].long()
    num_classes = int(labels_all.max().item()) + 1
    per_class = max(1, N_PROFILE // num_classes)
    profile_idx, eval_pool = stratified_profile_split(labels_all, per_class, seed=42)
    eval_idx = eval_pool[:N_EVAL]

    images = data["imgs"].float()
    labels = data["labels"].long()
    tokens = data["tokens"]

    profile_loader = DataLoader(
        CachedBirdSubset(images[profile_idx], labels[profile_idx]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=collate_cached,
    )
    eval_loader = DataLoader(
        CachedBirdSubset(images[eval_idx], labels[eval_idx]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        collate_fn=collate_cached,
    )
    profile_dino = [list(tokens[i]) for i in profile_idx]
    eval_dino = [list(tokens[i]) for i in eval_idx]
    return profile_loader, eval_loader, profile_dino, eval_dino, data["class_text_feats"].float()


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip = CLIPViT(model_name=MODEL_NAME).to(device)
    clip.eval()

    payload, cache_path, err = load_cache(root, MODEL_NAME, CACHE_SAMPLES)
    if payload is None:
        sys.exit(f"Cache unavailable: {err}\nExpected: {cache_path}\n"
                 f"Run precompute_birds_cache_perclass.py first.")
    meta = payload["meta"]
    detected = sum(payload["data"]["dino_detected"])
    total = len(payload["data"]["dino_detected"])
    print(f"Using cache: {cache_path}")
    print(f"Cached images: {meta['n_samples']}  |  query: per-class (VLM-generated)")
    print(f"DINO detected boxes on {detected}/{total} images ({100*detected/total:.1f}%)\n")

    profile_loader, eval_loader, profile_dino, eval_dino, class_text_feats = build_loaders(payload)
    class_text_feats = class_text_feats.to(device)

    print(f"Reserved profile split: {N_PROFILE} images")
    print(f"Eval split             : {N_EVAL} images")

    print(f"\nPhase 1: profiling layers {PROFILE_LAYERS} x {clip.num_heads} heads "
          f"on N={N_PROFILE} at α={PROFILE_ALPHA}...")
    ranked = profile_heads(
        clip, profile_loader, class_text_feats, profile_dino,
        alpha=PROFILE_ALPHA, layers=PROFILE_LAYERS,
    )
    print("Top-15 heads from profiling:")
    for (l, h), acc in ranked[:15]:
        print(f"  ({l:>2},{h:>2})  {acc:.2f}%")
    ranked_heads = [lh for lh, _ in ranked]

    print(f"\nPhase 2: alpha x head-count sweep on {N_EVAL} eval images")
    acc_plain = eval_plain(clip, eval_loader, class_text_feats)
    print(f"Plain CLIP: {acc_plain:.2f}%\n")
    grid = {}
    best = {"acc": acc_plain, "alpha": None, "heads": [], "head_count": 0}
    for alpha in PASTA_ALPHAS:
        for head_count in HEAD_COUNTS:
            chosen = ranked_heads[:min(head_count, len(ranked_heads))]
            acc = eval_pasta_cached(
                clip, eval_loader, class_text_feats, chosen, alpha, eval_dino,
            )
            grid[(alpha, head_count)] = acc
            if acc > best["acc"]:
                best = {"acc": acc, "alpha": alpha, "heads": chosen, "head_count": head_count}

    col_w = 11
    header = f"{'heads/alpha':<12}" + "".join(f"{a:>{col_w}.3f}" for a in PASTA_ALPHAS)
    print()
    print(header)
    print("-" * len(header))
    for hc in HEAD_COUNTS:
        row = f"{hc:<12}"
        for a in PASTA_ALPHAS:
            row += f"{grid[(a, hc)]:>{col_w-1}.2f} "
        print(row)

    print()
    print("Δ vs. plain CLIP:")
    print(header)
    print("-" * len(header))
    for hc in HEAD_COUNTS:
        row = f"{hc:<12}"
        for a in PASTA_ALPHAS:
            delta = grid[(a, hc)] - acc_plain
            row += f"{delta:>+{col_w-1}.2f} "
        print(row)

    print(f"\n{'='*60}")
    print(f"  Plain CLIP                                : {acc_plain:.2f}%")
    print(f"  Best PASTA (per-class) α={best['alpha']}, heads={best['head_count']:>2} : "
          f"{best['acc']:.2f}%  ({best['acc'] - acc_plain:+.2f}%)")
    print(f"  Heads used: {best['heads']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
