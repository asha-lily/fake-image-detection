"""Extract captions of real LAION images from an OpenFake shard, to use as generation prompts.

OpenFake's fakes were generated from captions of real LAION images, so reusing those captions as
prompts keeps the content of new test fakes matched to the real images.

Example (from a notebook in `notebooks/`):

    from pathlib import Path
    from fake_image_detection.prompts import extract_laion_prompts

    records = extract_laion_prompts(
        shard_name="train-00020-of-00032-00003",
        num_prompts=350,  # oversample: some prompts will be refused by some generators
        output_csv=Path("../data/test_prompts/prompts.csv"),
    )
"""

import csv
import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

OPENFAKE_SHARD_URL = "hf://datasets/ComplexDataLab/OpenFake/core/{shard_name}.parquet"

# Shards that train/test images were already taken from (see new_dataset_exploration.ipynb).
# Prompts should come from a different shard, so they aren't captions of real images that are
# already in the train/test sets.
USED_SHARDS = frozenset(
    {
        "train-00000-of-00032-00000",
        "train-00009-of-00032-00017",
        "train-00012-of-00032-00010",
        "train-00016-of-00032-00016",
        "train-00028-of-00032-00004",
    }
)

# Captions containing these words are skipped: providers refuse such prompts unevenly, which would
# leave each generator with a different subset of prompts. Extend this as needed.
BLOCKED_WORDS = frozenset({
    "nude", "naked", "nsfw", "sexy", "erotic", "lingerie", "topless", "porn",
    "blood", "gore", "corpse", "gun", "rifle", "pistol", "weapon",
    "logo", "trademark", "celebrity", "president",
})  # fmt: skip


@dataclass(frozen=True)
class PromptRecord:
    """A caption to use as a prompt, with where it came from.

    Attributes:
        prompt_id: unique ID, e.g. "s20-3_00412" (shard group 20, file 3, row 412).
        prompt: the caption text.
        source_shard: the OpenFake shard the caption came from.
        row_index: the caption's row in that shard.
    """

    prompt_id: str
    prompt: str
    source_shard: str
    row_index: int


def clean_caption(caption: str) -> str:
    """Collapse whitespace and strip surrounding quotes."""
    return " ".join(caption.split()).strip("\"' ")


def has_proper_noun_pair(caption: str) -> bool:
    """Heuristic for names of people, places or brands: two capitalised words in a row, not at the
    start of a sentence (e.g. "a portrait of Taylor Swift")."""
    words = caption.split()
    for i in range(1, len(words) - 1):
        starts_sentence = words[i - 1][-1] in ".!?"
        if not starts_sentence and words[i][:1].isupper() and words[i + 1][:1].isupper():
            return True
    return False


def is_usable_caption(
    caption: str,
    min_words: int,
    max_words: int,
    blocked_words: frozenset[str],
    drop_proper_nouns: bool,
) -> bool:
    """Whether a caption is suitable as a prompt for every generator."""
    words = caption.split()
    if not min_words <= len(words) <= max_words:
        return False
    # skip mostly non-English captions
    if sum(char.isascii() for char in caption) / len(caption) < 0.95:
        return False
    if {word.strip(".,;:!?()\"'").lower() for word in words} & blocked_words:
        return False
    return not (drop_proper_nouns and has_proper_noun_pair(caption))


def short_shard_id(shard_name: str) -> str:
    """Shorten a shard name, e.g. "train-00020-of-00032-00003" -> "s20-3"."""
    parts = shard_name.split("-")
    return f"s{int(parts[1])}-{int(parts[4])}"


def extract_laion_prompts(
    shard_name: str,
    num_prompts: int,
    output_csv: Path,
    min_words: int = 5,
    max_words: int = 100,
    blocked_words: frozenset[str] = BLOCKED_WORDS,
    drop_proper_nouns: bool = True,
) -> list[PromptRecord]:
    """Stream one OpenFake shard and collect captions of real LAION images as prompts.

    Args:
        shard_name: e.g. "train-00020-of-00032-00003". Must not be one of `USED_SHARDS`.
        num_prompts: how many prompts to collect. Collect more than you need (e.g. 350 for 300),
            since generators will refuse some prompts.
        output_csv: where to write the prompts.
        min_words: shortest caption to keep.
        max_words: longest caption to keep.
        blocked_words: captions containing any of these words are skipped.
        drop_proper_nouns: skip captions that look like they name a person, place or brand.

    Returns:
        The collected prompts, in shard order.
    """
    if shard_name in USED_SHARDS:
        raise ValueError(f"{shard_name} was already used for train/test images; pick another.")

    dataset = load_dataset(
        "parquet",
        data_files=OPENFAKE_SHARD_URL.format(shard_name=shard_name),
        split="train",
        streaming=True,
    ).select_columns(["prompt", "label", "model"])  # skip the images

    shard_id = short_shard_id(shard_name)
    records: list[PromptRecord] = []
    seen_captions: set[str] = set()

    for row_index, row in enumerate(tqdm(dataset, desc=shard_name)):
        if row["label"] != "real" or str(row["model"]).lower() != "laion":
            continue
        caption = clean_caption(row["prompt"] or "")
        if not caption or caption.lower() in seen_captions:
            continue
        if not is_usable_caption(caption, min_words, max_words, blocked_words, drop_proper_nouns):
            continue

        seen_captions.add(caption.lower())
        prompt_id = f"{shard_id}_{row_index:05d}"
        records.append(PromptRecord(prompt_id, caption, shard_name, row_index))
        if len(records) >= num_prompts:
            break

    if len(records) < num_prompts:
        logger.warning(f"Only found {len(records)} usable captions in {shard_name}.")

    write_prompts(records, output_csv)
    logger.info(f"Wrote {len(records)} prompts to {output_csv}")
    return records


def write_prompts(records: list[PromptRecord], path: Path) -> None:
    """Write prompts to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in fields(PromptRecord)])
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def load_prompts(path: Path) -> list[PromptRecord]:
    """Read prompts written by `write_prompts`."""
    with path.open(newline="") as file:
        return [
            PromptRecord(
                prompt_id=row["prompt_id"],
                prompt=row["prompt"],
                source_shard=row["source_shard"],
                row_index=int(row["row_index"]),
            )
            for row in csv.DictReader(file)
        ]
