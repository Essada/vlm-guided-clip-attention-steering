import base64
import os
from pathlib import Path

VQA_IMG_DIR = os.path.expanduser("~/data/vqa/images/val2014")

ICL_EXAMPLES = [
    {
        "image": os.path.join(VQA_IMG_DIR, "COCO_val2014_000000381037.jpg"),
        "question": "Can you see the cat's face?",
        "good": "cat . face . whiskers .",
        "bad": "animal . body . eyes .",
        "bad_reason": "too generic and avoids the parts that actually reveal whether the face is visible",
    },
    {
        "image": os.path.join(VQA_IMG_DIR, "COCO_val2014_000000449638.jpg"),
        "question": "Are there people on the street?",
        "good": "people . street . sidewalk .",
        "bad": "walking . cars .",
        "bad_reason": "names actions and unrelated objects instead of grounding the people and the street",
    },
]

SYSTEM_PROMPT = """\
Given a yes/no question about an image, list up to three concrete visible things to ground with Grounding DINO.

Rules:
- Output only one-word lowercase objects, parts, or regions.
- Separate words with periods, like: man . head . hat .
- Do not answer the question.
- Always output at least one visible thing. Never leave the answer blank.
- Choose visible evidence, not just words copied from the question.

Question: Is this a busy city?
Objects: people . cars . street .

Question: Is the room clean?
Objects: room . floor . table .

Question: Is the man wearing a hat?
Objects: man . head . hat .

Question: Is the person eating?
Objects: person . mouth . food ."""


def _encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_dino_prompt(question):
    examples_text = "\n\n".join(
        f"Question: {ex['question']}\nObjects: {ex['good']}" for ex in ICL_EXAMPLES
    )
    return f"""\
{SYSTEM_PROMPT}

{examples_text}

Question: {question}
Objects:"""


def build_dino_messages(question, image_path=None):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in ICL_EXAMPLES:
        if not Path(ex["image"]).exists():
            continue
        user_text = (
            f"Question: {ex['question']}\n"
            f"Bad output: {ex['bad']}  ({ex['bad_reason']})\n"
            "What objects should we ground?"
        )
        messages.append({
            "role": "user",
            "content": user_text,
            "images": [_encode_image(ex["image"])],
        })
        messages.append({"role": "assistant", "content": ex["good"]})

    target_user = {"role": "user", "content": f"Question: {question}\nObjects:"}
    if image_path and Path(image_path).exists():
        target_user["images"] = [_encode_image(image_path)]
    messages.append(target_user)
    return messages


def first_nonempty_line(text):
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def normalize_dino_query(text):
    query = first_nonempty_line(text).lower()
    if not query:
        return ""
    if not query.endswith("."):
        query = query + " ."
    return query
