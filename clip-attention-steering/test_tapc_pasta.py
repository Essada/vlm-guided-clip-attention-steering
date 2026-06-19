import os
import sys
import json
import torch
import requests
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from ConformationCLIP_ViT import CLIPViT
from test_tapc import (
    ollama,
    ollama_chat,
    is_yes_no_question,
    generate_yes_no_statements,
)
from tapc_grounding_utils import bbox_to_patch_indices_strict
from test_parts_steering import load_grounding_dino
from dino_prompt import build_dino_messages, normalize_dino_query

DATA_ROOT  = os.environ.get("VQA_DATA_ROOT", os.path.expanduser("~/data/vqa"))
N_PROFILE  = 500
N_EVAL     = 1000
BATCH_SIZE = 8
PRECOMPUTE_POOL_SIZE = int(os.environ.get("PRECOMPUTE_POOL_SIZE", "1500"))

PASTA_ALPHA    = 0.1
ALPHAS         = [0.02, 0.05, 0.1]
PROFILE_LAYERS = list(range(6, 12))
TOP_HEADS_K    = 5
HEAD_COUNTS    = [5, 10, 20]
MODEL_NAME     = os.environ.get("CLIP_MODEL", "ViT-B-16")

BOX_THRESHOLD  = 0.25
TEXT_THRESHOLD = 0.20
CACHE_DIR      = os.path.expanduser("~/data/vqa")
PRECOMPUTE_CACHE_VERSION = 1


def extract_dino_query(question: str, image_path: str = None) -> str:
    messages = build_dino_messages(question, image_path)
    return normalize_dino_query(ollama_chat(messages)) or "image ."


def dino_patch_indices(pil_img, gd_proc, gd_model, device, query, patch_size, grid_size):
    orig_w, orig_h = pil_img.size
    inputs = gd_proc(images=pil_img, text=query, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    try:
        results = gd_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]
    except TypeError:
        results = gd_proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]

    boxes = results["boxes"]
    if boxes.shape[0] == 0:
        return [], False

    indices = set()
    for box in boxes.cpu().tolist():
        indices.update(bbox_to_patch_indices_strict(box, orig_w, orig_h, patch_size=patch_size, grid_size=grid_size))
    return (list(indices), True) if indices else ([], False)


class YesNoVQADataset(Dataset):
    def __init__(self, root, transform):
        q_path  = os.path.join(root, "v2_OpenEnded_mscoco_val2014_questions.json")
        a_path  = os.path.join(root, "v2_mscoco_val2014_annotations.json")
        img_dir = os.path.join(root, "images", "val2014")

        with open(q_path) as f:
            questions = {q["question_id"]: q for q in json.load(f)["questions"]}
        with open(a_path) as f:
            annotations = json.load(f)["annotations"]

        self.samples = []
        for ann in annotations:
            answer = ann["multiple_choice_answer"].lower()
            if answer not in ("yes", "no"):
                continue
            q = questions[ann["question_id"]]
            if not is_yes_no_question(q["question"]):
                continue
            img_path = os.path.join(img_dir, f"COCO_val2014_{q['image_id']:012d}.jpg")
            if not os.path.exists(img_path):
                continue
            self.samples.append({
                "cache_key":   f"{ann['question_id']}:{q['image_id']}",
                "question_id": ann["question_id"],
                "image_id":    q["image_id"],
                "question":   q["question"],
                "answer":     answer,
                "image_path": img_path,
            })

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        pil = Image.open(s["image_path"]).convert("RGB")
        return (
            self.transform(pil),
            s["answer"],
            s["question"],
            pil,
            s["cache_key"],
            s["image_path"],
        )


def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    answers = [b[1] for b in batch]
    qs      = [b[2] for b in batch]
    pils    = [b[3] for b in batch]
    keys    = [b[4] for b in batch]
    paths   = [b[5] for b in batch]
    return imgs, answers, qs, pils, keys, paths


def subset_data(data, start, end):
    return {
        "imgs": data["imgs"][start:end],
        "tokens": data["tokens"][start:end],
        "dino_detected": data["dino_detected"][start:end],
        "yes_feats": data["yes_feats"][start:end],
        "no_feats": data["no_feats"][start:end],
        "gts": data["gts"][start:end],
        "questions": data["questions"][start:end],
        "keys": data["keys"][start:end],
        "image_paths": data["image_paths"][start:end],
    }


def _sanitize_model_name(name: str) -> str:
    return name.lower().replace("/", "-").replace(" ", "-")


def get_precompute_cache_path(model_name: str, pool_size: int) -> str:
    filename = f"tapc_pasta_precompute_{_sanitize_model_name(model_name)}_{pool_size}.pt"
    return os.path.join(CACHE_DIR, filename)


def load_precomputed_data(model_name: str, pool_size: int, root: str):
    cache_path = get_precompute_cache_path(model_name, pool_size)
    if not os.path.exists(cache_path):
        return None, cache_path

    payload = torch.load(cache_path, map_location="cpu")
    meta = payload.get("meta", {})
    if meta.get("cache_version") != PRECOMPUTE_CACHE_VERSION:
        return None, cache_path
    if meta.get("model_name") != model_name or meta.get("pool_size") != pool_size:
        return None, cache_path
    if meta.get("data_root") != os.path.abspath(root):
        return None, cache_path
    if meta.get("box_threshold") != BOX_THRESHOLD or meta.get("text_threshold") != TEXT_THRESHOLD:
        return None, cache_path
    return payload["data"], cache_path


def save_precomputed_data(data, model_name: str, pool_size: int, clip, cache_path: str, root: str):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    payload = {
        "meta": {
            "cache_version": PRECOMPUTE_CACHE_VERSION,
            "model_name": model_name,
            "pool_size": pool_size,
            "patch_size": clip.patch_size,
            "grid_size": clip.grid_size,
            "num_spatial_tokens": clip.num_spatial_tokens,
            "data_root": os.path.abspath(root),
            "box_threshold": BOX_THRESHOLD,
            "text_threshold": TEXT_THRESHOLD,
        },
        "data": data,
    }
    torch.save(payload, cache_path)


def precompute(clip, loader, gd_proc, gd_model, device):
    imgs_all, gts_all, toks_all, detected_all = [], [], [], []
    qs_all, keys_all, paths_all = [], [], []
    yes_feats, no_feats = [], []

    for imgs, gts, qs, pils, keys, paths in tqdm(loader, desc="Precompute"):
        imgs_all.append(imgs)
        gts_all.extend(gts)
        qs_all.extend(qs)
        keys_all.extend(keys)
        paths_all.extend(paths)
        for q, pil, key, path in zip(qs, pils, keys, paths):
            dino_query = extract_dino_query(q, image_path=path)
            yes_stmt, no_stmt = generate_yes_no_statements(q)

            toks, detected = dino_patch_indices(
                pil, gd_proc, gd_model, device, dino_query,
                patch_size=clip.patch_size, grid_size=clip.grid_size,
            )
            toks_all.append(toks)
            detected_all.append(detected)
            with torch.no_grad():
                tf = clip.encode_text([yes_stmt, no_stmt])
            yes_feats.append(tf[0].cpu())
            no_feats.append(tf[1].cpu())

    return {
        "imgs":      torch.cat(imgs_all),
        "tokens":    toks_all,
        "dino_detected": detected_all,
        "yes_feats": torch.stack(yes_feats),
        "no_feats":  torch.stack(no_feats),
        "gts":       gts_all,
        "questions":  qs_all,
        "keys":       keys_all,
        "image_paths": paths_all,
    }


def _score_yes_no(feats, yes_feats, no_feats, gts):
    sy   = (feats * yes_feats).sum(dim=-1)
    sn   = (feats * no_feats).sum(dim=-1)
    pred = torch.where(sy >= sn, 1, 0).tolist()
    pred_labels = ["yes" if p == 1 else "no" for p in pred]
    correct = sum(
        (p == 1 and g == "yes") or (p == 0 and g == "no")
        for p, g in zip(pred, gts)
    )
    return correct, pred_labels


def eval_plain(clip, data, batch_size=BATCH_SIZE, return_preds=False):
    dev = next(clip.parameters()).device
    N   = data["imgs"].shape[0]
    correct = 0
    preds = []
    for i in range(0, N, batch_size):
        imgs = data["imgs"][i:i+batch_size].to(dev)
        with torch.no_grad():
            feats = clip.encode_image(imgs).cpu()
        batch_correct, batch_preds = _score_yes_no(
            feats,
            data["yes_feats"][i:i+batch_size],
            data["no_feats"][i:i+batch_size],
            data["gts"][i:i+batch_size],
        )
        correct += batch_correct
        preds.extend(batch_preds)
    acc = correct / N * 100
    return (acc, preds) if return_preds else acc


def eval_pasta(clip, data, heads, alpha, batch_size=BATCH_SIZE, desc="PASTA",
               return_preds=False):
    dev = next(clip.parameters()).device
    N   = data["imgs"].shape[0]
    correct = 0
    preds = []
    for i in range(0, N, batch_size):
        imgs = data["imgs"][i:i+batch_size].to(dev)
        toks = data["tokens"][i:i+batch_size]
        with torch.no_grad():
            feats = clip.encode_image_pasta(imgs, toks, alpha, heads).cpu()
        batch_correct, batch_preds = _score_yes_no(
            feats,
            data["yes_feats"][i:i+batch_size],
            data["no_feats"][i:i+batch_size],
            data["gts"][i:i+batch_size],
        )
        correct += batch_correct
        preds.extend(batch_preds)
    acc = correct / N * 100
    return (acc, preds) if return_preds else acc


def save_eval_results(data, plain_preds, pasta_preds, heads, alpha):
    results = []
    for i, key in enumerate(data["keys"]):
        gt = data["gts"][i]
        results.append({
            "cache_key": key,
            "question": data["questions"][i],
            "image_path": data["image_paths"][i],
            "gt": gt,
            "plain_pred": plain_preds[i],
            "plain_correct": plain_preds[i] == gt,
            "pasta_pred": pasta_preds[i],
            "pasta_correct": pasta_preds[i] == gt,
            "pred": pasta_preds[i],
            "correct": pasta_preds[i] == gt,
            "is_yes_no": True,
            "pasta_alpha": alpha,
            "pasta_heads": [[l, h] for l, h in heads],
            "dino_detected": data["dino_detected"][i],
            "token_count": len(data["tokens"][i]),
            "tokens": data["tokens"][i],
        })
    return results


def profile_heads(clip, data, layers, alpha, batch_size=BATCH_SIZE):
    scores = {}
    pairs  = [(l, h) for l in layers for h in range(clip.num_heads)]
    for l, h in tqdm(pairs, desc="Profiling heads"):
        acc = eval_pasta(clip, data, [(l, h)], alpha, batch_size=batch_size)
        scores[(l, h)] = acc
    return sorted(scores.items(), key=lambda x: -x[1])


def main():
    root   = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    device = "cpu"

    clip = CLIPViT(model_name=MODEL_NAME).to(device)
    clip.eval()

    total_needed = N_PROFILE + N_EVAL
    if total_needed > PRECOMPUTE_POOL_SIZE:
        print(
            f"ERROR: profile+eval requires {total_needed} samples, "
            f"but PRECOMPUTE_POOL_SIZE={PRECOMPUTE_POOL_SIZE}."
        )
        sys.exit(1)

    print(
        f"Model: {MODEL_NAME}  grid={clip.grid_size}x{clip.grid_size}  patch={clip.patch_size}\n"
        f"Loading VQAv2 yes/no subset (profile={N_PROFILE}, eval={N_EVAL}, "
        f"precompute_pool={PRECOMPUTE_POOL_SIZE}, total_needed={total_needed})..."
    )
    data, cache_path = load_precomputed_data(MODEL_NAME, PRECOMPUTE_POOL_SIZE, root)
    if data is None:
        try:
            requests.get("http://localhost:11434", timeout=3)
        except Exception:
            print("ERROR: Ollama is not running. Start with: ollama serve")
            sys.exit(1)
        gd_proc, gd_model = load_grounding_dino(device)
        full_ds = YesNoVQADataset(root, clip.preprocess)
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(full_ds), generator=g)[:PRECOMPUTE_POOL_SIZE].tolist()
        subset = torch.utils.data.Subset(full_ds, idx)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)
        print(f"Loaded {len(subset)} samples for precompute")
        print(f"Precomputing and saving shared pool -> {cache_path}\n")
        data = precompute(clip, loader, gd_proc, gd_model, device)
        save_precomputed_data(data, MODEL_NAME, PRECOMPUTE_POOL_SIZE, clip, cache_path, root)
    else:
        print(f"Loaded shared precompute pool from {cache_path}\n")

    data_profile = subset_data(data, 0, N_PROFILE)
    data_eval = subset_data(data, N_PROFILE, N_PROFILE + N_EVAL)

    acc_plain_prof = eval_plain(clip, data_profile)
    print(f"\nPlain TAP-C (profile, N={N_PROFILE}): {acc_plain_prof:.2f}%\n")

    ranked = profile_heads(clip, data_profile, PROFILE_LAYERS, PASTA_ALPHA)

    print(f"\nTop-10 (layer, head) pairs — profiled on N={N_PROFILE}:")
    print(f"  {'(layer,head)':<14} {'acc':>8}  {'Δ':>8}")
    for (l, h), acc in ranked[:10]:
        print(f"  ({l},{h}){'':<10} {acc:>8.2f}  {acc - acc_plain_prof:>+8.2f}%")

    ranked_heads = [lh for lh, _ in ranked]
    best_heads = [lh for lh, _ in ranked[:TOP_HEADS_K]]
    print(f"\nSelected top-{TOP_HEADS_K} heads: {best_heads}")

    acc_plain, plain_preds = eval_plain(clip, data_eval, return_preds=True)

    print()
    print(f"Head-count and alpha sweep on N={N_EVAL}:")
    print("  heads   alpha        acc  delta vs plain")
    print("  " + "-" * 44)

    best_alpha = None
    best_acc = acc_plain
    best_preds = plain_preds
    best_heads_for_eval = []
    sweep_results = []

    for head_count in HEAD_COUNTS:
        chosen_heads = ranked_heads[:min(head_count, len(ranked_heads))]
        best_for_k = {
            "alpha": None,
            "acc": acc_plain,
            "preds": plain_preds,
            "heads": chosen_heads,
        }

        for alpha in ALPHAS:
            acc_pasta, pasta_preds = eval_pasta(
                clip, data_eval, chosen_heads, alpha,
                desc=f"PASTA k={len(chosen_heads)} alpha={alpha}", return_preds=True,
            )
            sweep_results.append((len(chosen_heads), alpha, acc_pasta))
            print(f"  {len(chosen_heads):<7} {alpha:<8} {acc_pasta:>8.2f}  {acc_pasta - acc_plain:>+10.2f}%")
            if acc_pasta > best_for_k["acc"]:
                best_for_k = {
                    "alpha": alpha,
                    "acc": acc_pasta,
                    "preds": pasta_preds,
                    "heads": chosen_heads,
                }
            if acc_pasta > best_acc:
                best_alpha = alpha
                best_acc = acc_pasta
                best_preds = pasta_preds
                best_heads_for_eval = chosen_heads

        if best_for_k["alpha"] is None:
            print(f"    best for top-{len(chosen_heads)}: plain baseline")
        else:
            print(
                f"    best for top-{len(chosen_heads)}: alpha={best_for_k['alpha']} "
                f"acc={best_for_k['acc']:.2f}%  ({best_for_k['acc'] - acc_plain:+.2f}%)"
            )

    results = save_eval_results(data_eval, plain_preds, best_preds, best_heads_for_eval, best_alpha)

    print()
    print("=" * 60)
    print(f"  Plain TAP-C (N={N_EVAL})              : {acc_plain:.2f}%")
    if best_alpha is None:
        print("  Best PASTA TAP-C                      : no config beat plain")
        print("  Result records kept in memory         : plain baseline")
    else:
        print(f"  Best PASTA TAP-C alpha={best_alpha:<6} : {best_acc:.2f}%  ({best_acc - acc_plain:+.2f}%)")
        print(f"  Heads used ({len(best_heads_for_eval)}): {best_heads_for_eval}")
        print(f"  Result records kept in memory         : {len(results)}")
    print(f"  Sweep results: {sweep_results}")
    print("=" * 60)


if __name__ == "__main__":
    main()
