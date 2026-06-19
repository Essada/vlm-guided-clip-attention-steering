import os
import sys
import json
import re
import torch
import requests
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from ConformationCLIP_ViT import CLIPViT

DATA_ROOT    = os.environ.get("VQA_DATA_ROOT", os.path.expanduser("~/data/vqa"))
N_EVAL       = 100
BATCH_SIZE   = 8
TOP_K_ANS    = 1000
FILTER_K     = 20
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma4:e4b"
CACHE_FILE   = os.path.expanduser("~/data/vqa/tapc_cache.json")


def ollama(prompt: str, temperature: float = 0.0) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 200},
    }, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    if "message" in data:
        return data["message"].get("content", "").strip()
    return data.get("response", "").strip()


def ollama_chat(messages, temperature: float = 0.0) -> str:
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 200},
    }, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    if "message" in data:
        return data["message"].get("content", "").strip()
    return data.get("response", "").strip()


YES_NO_STARTERS = (
    "is ", "are ", "was ", "were ", "do ", "does ", "did ",
    "can ", "could ", "will ", "would ", "has ", "have ", "had ",
    "should ", "may ", "might ",
)


def is_yes_no_question(question: str) -> bool:
    return question.strip().lower().startswith(YES_NO_STARTERS)


def generate_template(question: str) -> str:
    prompt = (
        "Convert each question into a fill-in-the-blank declarative statement. "
        "Output only the statement ending with a blank, nothing else.\n\n"
        "Question: What color is the car?\n"
        "Statement: The color of the car is\n\n"
        "Question: How many dogs are there?\n"
        "Statement: There are ___ dogs\n\n"
        "Question: What is the person doing?\n"
        "Statement: The person is\n\n"
        "Question: What sport is being played?\n"
        "Statement: The sport being played is\n\n"
        f"Question: {question}\n"
        "Statement:"
    )
    result = ollama(prompt)
    return result.split("\n")[0].strip()


def generate_yes_no_statements(question: str) -> tuple:
    prompt = (
        "Convert each yes/no question into two declarative statements: one "
        "affirmative (assuming the answer is yes) and one negative (assuming no). "
        "Output exactly two lines:\n"
        "YES: <affirmative statement>\n"
        "NO: <negative statement>\n\n"
        "Question: Is the sky blue?\n"
        "YES: The sky is blue.\n"
        "NO: The sky is not blue.\n\n"
        "Question: Are the dogs playing?\n"
        "YES: The dogs are playing.\n"
        "NO: The dogs are not playing.\n\n"
        "Question: Does the man have a hat?\n"
        "YES: The man has a hat.\n"
        "NO: The man does not have a hat.\n\n"
        f"Question: {question}\n"
    )
    result = ollama(prompt)

    yes, no = "", ""
    for line in result.split("\n"):
        line = line.strip()
        if line.upper().startswith("YES:"):
            yes = line[4:].strip()
        elif line.upper().startswith("NO:"):
            no = line[3:].strip()

    if not yes or not no:
        base = question.rstrip("?").strip()
        yes  = base
        no   = f"it is not true that {base}"

    return yes, no


def filter_answers(template: str, answers: list, top_k: int = FILTER_K) -> list:
    answers_str = ", ".join(answers[:200])
    prompt = (
        f"Given the statement template: \"{template} ___\"\n"
        f"From this list of candidate answers, return only the {top_k} most "
        f"plausible ones that could fill the blank. "
        f"Return them as a comma-separated list, nothing else.\n\n"
        f"Candidates: {answers_str}\n\n"
        f"Top {top_k} most plausible:"
    )
    result = ollama(prompt)

    filtered = [a.strip().lower().rstrip(".,") for a in result.split(",")]
    answer_set = {a.lower() for a in answers}
    filtered   = [a for a in filtered if a in answer_set]

    if len(filtered) < 3:
        return answers[:top_k]
    return filtered[:top_k]


class VQADataset(Dataset):
    def __init__(self, root, transform, top_k_answers=TOP_K_ANS, n_eval=N_EVAL):
        q_path  = os.path.join(root, "v2_OpenEnded_mscoco_val2014_questions.json")
        a_path  = os.path.join(root, "v2_mscoco_val2014_annotations.json")
        img_dir = os.path.join(root, "images", "val2014")

        with open(q_path) as f:
            questions = {q["question_id"]: q for q in json.load(f)["questions"]}
        with open(a_path) as f:
            annotations = json.load(f)["annotations"]

        all_answers  = [a["multiple_choice_answer"] for a in annotations]
        self.answers = [ans for ans, _ in Counter(all_answers).most_common(top_k_answers)]
        answer_set   = set(self.answers)

        self.samples = []
        for ann in annotations:
            qid    = ann["question_id"]
            answer = ann["multiple_choice_answer"]
            if answer not in answer_set:
                continue
            q        = questions[qid]
            img_path = os.path.join(img_dir, f"COCO_val2014_{q['image_id']:012d}.jpg")
            if not os.path.exists(img_path):
                continue
            self.samples.append({
                "cache_key":  f"{qid}:{q['image_id']}",
                "question_id": qid,
                "image_id":    q["image_id"],
                "question":   q["question"],
                "answer":     answer,
                "image_path": img_path,
            })

        g   = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(self.samples), generator=g)[:n_eval].tolist()
        self.samples  = [self.samples[i] for i in idx]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["image_path"]).convert("RGB")
        return self.transform(img), s["answer"], s["question"], s["cache_key"]


def collate_fn(batch):
    imgs    = torch.stack([b[0] for b in batch])
    answers = [b[1] for b in batch]
    qs      = [b[2] for b in batch]
    keys    = [b[3] for b in batch]
    return imgs, answers, qs, keys


@torch.no_grad()
def eval_qip(model, loader):
    dev = next(model.parameters()).device
    correct, total = 0, 0
    all_answers = loader.dataset.answers

    for imgs, gt_answers, questions, _ in tqdm(loader, desc="QIP"):
        imgs  = imgs.to(dev)
        feats = model.encode_image(imgs)

        for i, q in enumerate(questions):
            prompts = [f"question: {q} answer: {a}" for a in all_answers]
            t_feats = model.encode_text(prompts)
            pred    = all_answers[(feats[i] @ t_feats.T).argmax().item()]
            if pred.lower() == gt_answers[i].lower():
                correct += 1
        total += imgs.shape[0]

    return correct / total * 100


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


@torch.no_grad()
def eval_tapc(model, loader):
    dev         = next(model.parameters()).device
    correct, total = 0, 0
    all_answers = loader.dataset.answers
    cache       = load_cache()
    cache_dirty = False

    for imgs, gt_answers, questions, cache_keys in tqdm(loader, desc="TAP-C"):
        imgs  = imgs.to(dev)
        feats = model.encode_image(imgs)

        for i, q in enumerate(questions):
            entry = cache.get(cache_keys[i], {})
            entry["question"] = q

            if is_yes_no_question(q):
                if "yes_stmt" not in entry or "no_stmt" not in entry:
                    yes_stmt, no_stmt  = generate_yes_no_statements(q)
                    entry["yes_stmt"]  = yes_stmt
                    entry["no_stmt"]   = no_stmt
                    entry["is_yes_no"] = True
                    cache_dirty = True
                else:
                    yes_stmt, no_stmt = entry["yes_stmt"], entry["no_stmt"]

                t_feats = model.encode_text([yes_stmt, no_stmt])
                scores  = feats[i] @ t_feats.T
                pred    = "yes" if scores[0].item() >= scores[1].item() else "no"

            else:
                if "template" not in entry:
                    entry["template"]  = generate_template(q)
                    entry["is_yes_no"] = False
                    cache_dirty = True
                template = entry["template"]

                if "candidates" not in entry or not entry["candidates"]:
                    entry["candidates"] = filter_answers(template, all_answers)
                    cache_dirty = True
                candidates = entry["candidates"]

                prompts = [f"{template} {a}" for a in candidates]
                t_feats = model.encode_text(prompts)
                pred    = candidates[(feats[i] @ t_feats.T).argmax().item()]

            match = pred.lower() == gt_answers[i].lower()
            entry["gt"]      = gt_answers[i]
            entry["pred"]    = pred
            entry["correct"] = match
            cache[cache_keys[i]] = entry
            cache_dirty      = True

            if match:
                correct += 1

        total += imgs.shape[0]

        if cache_dirty:
            save_cache(cache)
            cache_dirty = False

    return correct / total * 100


def main():
    root   = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    device = "cpu"

    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        print("ERROR: Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    clip = CLIPViT().to(device)
    clip.eval()

    print(f"Loading VQAv2 ({N_EVAL} samples, top-{TOP_K_ANS} answers)...")
    ds     = VQADataset(root, clip.preprocess)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)
    print(f"Loaded {len(ds)} samples\n")

    acc_tapc = eval_tapc(clip, loader)

    print(f"\n{'='*45}")
    print(f"  TAP-C (Qwen2.5)  : {acc_tapc:.2f}%")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
