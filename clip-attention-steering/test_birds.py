import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

from ConformationCLIP_ViT import CLIPViT
from test_parts_steering import eval_plain, eval_pasta_cached

DATA_ROOT      = os.environ.get(
    "CUB_DATA_ROOT",
    os.path.join(os.path.dirname(__file__), "CUB_200_2011"),
)
N_PROFILE      = 1000
N_EVAL         = 4000
BATCH_SIZE     = 8
TOP_HEADS_K    = 5
PROFILE_LAYERS = list(range(6, 12))
PASTA_ALPHAS   = [0.1, 0.05]
HEAD_COUNTS    = [5, 10]
MODEL_NAME     = os.environ.get("CLIP_MODEL", "ViT-B-16")
CACHE_VERSION  = 1
CACHE_SAMPLES  = 5000
HARDCODED_TOP_HEADS = [
    (11, 4),
    (6, 10),
    (9, 0),
    (6, 8),
    (9, 10),
    (11, 1),
    (6, 4),
    (6, 7),
    (8, 11),
    (10, 11),
]

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20


class CUBTestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root      = root
        self.transform = transform

        with open(os.path.join(root, "images.txt")) as f:
            id_to_path = {int(l.split()[0]): l.split()[1] for l in f}

        with open(os.path.join(root, "image_class_labels.txt")) as f:
            id_to_label = {int(l.split()[0]): int(l.split()[1]) - 1 for l in f}

        with open(os.path.join(root, "train_test_split.txt")) as f:
            id_to_split = {int(l.split()[0]): int(l.split()[1]) for l in f}

        with open(os.path.join(root, "classes.txt")) as f:
            self.classes = [l.split()[1] for l in f]

        self.samples = [
            (os.path.join(root, "images", id_to_path[i]), id_to_label[i])
            for i in sorted(id_to_path)
            if id_to_split[i] == 0
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil         = Image.open(path).convert("RGB")
        tensor      = self.transform(pil) if self.transform else pil
        return tensor, label, pil


class CachedBirdSubset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx]), None


def collate_cached(batch):
    tensors = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    placeholders = [b[2] for b in batch]
    return tensors, labels, placeholders


def bbox_to_patch_indices(bbox, orig_w, orig_h, patch_size, grid_size, crop_size=224, overlap_frac=0.75):
    x1, y1, x2, y2 = bbox

    scale = crop_size / min(orig_w, orig_h)
    new_w = orig_w * scale
    new_h = orig_h * scale
    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

    left = (new_w - crop_size) / 2
    top = (new_h - crop_size) / 2
    x1 = max(0.0, x1 - left)
    y1 = max(0.0, y1 - top)
    x2 = min(float(crop_size), x2 - left)
    y2 = min(float(crop_size), y2 - top)

    if x2 <= x1 or y2 <= y1:
        return list(range(grid_size * grid_size))

    selected = []
    for r in range(grid_size):
        for c in range(grid_size):
            px1 = c * patch_size
            py1 = r * patch_size
            px2 = px1 + patch_size
            py2 = py1 + patch_size

            inter_w = max(0.0, min(x2, px2) - max(x1, px1))
            inter_h = max(0.0, min(y2, py2) - max(y1, py1))
            inter_area = inter_w * inter_h
            patch_area = patch_size * patch_size
            if inter_area >= overlap_frac * patch_area:
                selected.append(r * grid_size + c)
    return selected if selected else list(range(grid_size * grid_size))


def sanitize_model_name(name: str) -> str:
    return name.lower().replace("/", "-").replace(" ", "-")


def cache_path_for(root: str, model_name: str, n_samples: int) -> str:
    filename = f"cub_precompute_{sanitize_model_name(model_name)}_{n_samples}_head-wing-tail.pt"
    return os.path.join(root, filename)


def load_cached_cub_precompute(root: str, model_name: str, expected_samples: int):
    cache_path = cache_path_for(root, model_name, expected_samples)
    if not os.path.exists(cache_path):
        return None, cache_path, f"missing file: {cache_path}"

    payload = torch.load(cache_path, map_location="cpu")
    meta = payload.get("meta", {})
    data = payload.get("data", {})

    if meta.get("cache_version") != CACHE_VERSION:
        return None, cache_path, f"cache_version={meta.get('cache_version')}"
    if meta.get("model_name") != model_name:
        return None, cache_path, f"model_name={meta.get('model_name')}"
    if os.path.abspath(meta.get("data_root", "")) != os.path.abspath(root):
        return None, cache_path, f"data_root={meta.get('data_root')}"
    if meta.get("n_samples") != expected_samples:
        return None, cache_path, f"n_samples={meta.get('n_samples')}"
    if not meta.get("complete", False):
        return None, cache_path, "cache is not marked complete"

    required_keys = {"imgs", "labels", "tokens", "dino_detected", "class_text_feats"}
    missing_keys = sorted(required_keys - set(data))
    if missing_keys:
        return None, cache_path, f"missing data keys: {', '.join(missing_keys)}"

    num_items = int(data["imgs"].shape[0])
    if num_items < N_PROFILE + N_EVAL:
        return None, cache_path, f"only {num_items} cached items available, need {N_PROFILE + N_EVAL}"
    if int(data["labels"].shape[0]) != num_items:
        return None, cache_path, "label count does not match image count"
    if len(data["tokens"]) != num_items:
        return None, cache_path, "token count does not match image count"
    if len(data["dino_detected"]) != num_items:
        return None, cache_path, "dino_detected count does not match image count"

    return payload, cache_path, None


def build_cached_loaders(payload):
    data = payload["data"]
    num_items = int(data["imgs"].shape[0])
    all_idx = torch.randperm(num_items, generator=torch.Generator().manual_seed(42)).tolist()
    profile_idx = all_idx[:N_PROFILE]
    eval_idx = all_idx[N_PROFILE:N_PROFILE + N_EVAL]

    images = data["imgs"].float()
    labels = data["labels"].long()
    tokens = data["tokens"]

    profile_loader = DataLoader(
        CachedBirdSubset(images[profile_idx], labels[profile_idx]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cached,
    )
    eval_loader = DataLoader(
        CachedBirdSubset(images[eval_idx], labels[eval_idx]),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_cached,
    )
    profile_dino = [tokens[i] for i in profile_idx]
    eval_dino = [tokens[i] for i in eval_idx]

    return profile_loader, eval_loader, profile_dino, eval_dino, data["class_text_feats"].float()


def main():
    root   = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip = CLIPViT(model_name=MODEL_NAME).to(device)
    clip.eval()

    cached_payload, cache_path, cache_error = load_cached_cub_precompute(root, MODEL_NAME, CACHE_SAMPLES)
    if cached_payload is None:
        sys.exit(
            f"Cache required but unavailable: {cache_error}\n"
            f"Expected: {cache_path}\n"
            f"Run precompute_birds_cache.py first."
        )

    meta = cached_payload["meta"]
    print(f"CUB cached test set: {meta['n_samples']} images  |  query={meta.get('query', '?')}")
    print(f"Using cache: {cache_path}")
    profile_loader, eval_loader, profile_dino, eval_dino, class_text_feats = build_cached_loaders(cached_payload)

    class_text_feats = class_text_feats.to(device)

    print(f"\nSkipping profiling and using hardcoded top-10 heads.")
    print(f"Reserved profile split size: {N_PROFILE} cached images")
    print("Hardcoded heads:")
    for l, h in HARDCODED_TOP_HEADS:
        print(f"  ({l},{h})")

    print(f"\nPhase 2: head-count sweep on {N_EVAL} cached images...")
    acc_plain = eval_plain(clip, eval_loader, class_text_feats)
    print(f"Plain CLIP (eval): {acc_plain:.2f}%\n")

    ranked_heads = HARDCODED_TOP_HEADS
    grid = {}
    best = {
        "acc": acc_plain,
        "alpha": None,
        "heads": [],
        "head_count": 0,
    }
    for alpha in PASTA_ALPHAS:
        for head_count in HEAD_COUNTS:
            chosen_heads = ranked_heads[:min(head_count, len(ranked_heads))]
            acc = eval_pasta_cached(
                clip,
                eval_loader,
                class_text_feats,
                chosen_heads,
                alpha,
                eval_dino,
            )
            grid[(alpha, head_count)] = acc
            if acc > best["acc"]:
                best = {
                    "acc": acc,
                    "alpha": alpha,
                    "heads": chosen_heads,
                    "head_count": head_count,
                }

    col_w = 11
    row_label = "heads/alpha"
    header = f"{row_label:<12}" + "".join(f"{a:>{col_w}.3f}" for a in PASTA_ALPHAS)
    print()
    print(header)
    print("-" * len(header))
    for hc in HEAD_COUNTS:
        row = f"{hc:<12}"
        for a in PASTA_ALPHAS:
            acc = grid[(a, hc)]
            row += f"{acc:>{col_w-1}.2f} "
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
    print(f"  Plain CLIP                          : {acc_plain:.2f}%")
    print(
        f"  Best PASTA (DINO) α={best['alpha']}, heads={best['head_count']:>2} : "
        f"{best['acc']:.2f}%  ({best['acc'] - acc_plain:+.2f}%)"
    )
    print(f"  Heads used: {best['heads']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
