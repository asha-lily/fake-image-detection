"""Generate test images from prompts with hosted image-generation APIs.

Each provider function sends one prompt and returns the image as bytes. `generate_images` loops
over prompts and generators, saves every image as PNG, and appends a row per attempt to a manifest
CSV. Prompts already done (or refused) are skipped, so an interrupted run can simply be restarted.

API keys are read from environment variables, never from code:
    OPENAI_API_KEY, GEMINI_API_KEY, DASHSCOPE_API_KEY

Model IDs and request parameters change often. Check `GENERATORS` against each provider's docs
before a full run, and try a few prompts first:

    python -m fake_image_detection.generate --generators gpt_image_2 --limit 3
"""

import argparse
import base64
import csv
import io
import json
import logging
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image
from tqdm import tqdm

from fake_image_detection.prompts import PromptRecord, load_prompts

logger = logging.getLogger(__name__)

# Alibaba Cloud's international (Singapore) endpoint. API keys are region-specific.
DASHSCOPE_BASE_URL = "https://dashscope-intl.aliyuncs.com/api/v1"

MANIFEST_FIELDS = [
    "prompt_id", "generator", "model_id", "status", "filename", "width", "height",
    "params", "error", "prompt", "timestamp",
]  # fmt: skip


class GenerationRefused(Exception):
    """The provider declined to generate an image for a prompt, e.g. because of a safety filter."""


@dataclass(frozen=True)
class Generator:
    """An image generator and the settings used to call it.

    Attributes:
        name: folder name for its images, e.g. "gpt_image_2" -> data/test/mllm/gpt_image_2.
        model_id: the provider's model ID. Prefer a dated snapshot, for reproducibility.
        params: request parameters, recorded in the manifest with every image.
        generate: function (prompt, model_id, params) -> image bytes.
    """

    name: str
    model_id: str
    params: dict[str, Any]
    generate: Callable[[str, str, dict[str, Any]], bytes]


# --- OpenAI (GPT Image) ----------------------------------------------------------------------


@cache
def _openai_client() -> Any:  # noqa: ANN401
    from openai import OpenAI  # imported here so you only need the SDKs you use

    return OpenAI()  # reads OPENAI_API_KEY


def generate_openai(prompt: str, model_id: str, params: dict[str, Any]) -> bytes:
    """Generate an image with OpenAI's Images API. GPT Image models return base64 PNG data."""
    from openai import BadRequestError

    try:
        result = _openai_client().images.generate(model=model_id, prompt=prompt, n=1, **params)
    except BadRequestError as error:
        if "moderation" in str(error).lower() or "safety" in str(error).lower():
            raise GenerationRefused(str(error)) from error
        raise
    return base64.b64decode(result.data[0].b64_json)


# --- Google (Nano Banana / Gemini image models) ----------------------------------------------


@cache
def _gemini_client() -> Any:  # noqa: ANN401
    from google import genai

    return genai.Client()  # reads GEMINI_API_KEY


def generate_gemini(prompt: str, model_id: str, params: dict[str, Any]) -> bytes:
    """Generate an image with the Gemini API's generate_content endpoint."""
    from google.genai import types

    response = _gemini_client().models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=params["aspect_ratio"]),
        ),
    )
    for candidate in response.candidates or []:
        parts = candidate.content.parts if candidate.content else None
        for part in parts or []:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data
    # no image: usually a safety block, reported in prompt_feedback or the finish reason
    finish_reasons = [str(candidate.finish_reason) for candidate in response.candidates or []]
    raise GenerationRefused(
        f"No image returned. prompt_feedback={response.prompt_feedback}, "
        f"finish_reasons={finish_reasons}"
    )


# --- Alibaba Cloud Model Studio (Qwen-Image) -------------------------------------------------


def generate_qwen(prompt: str, model_id: str, params: dict[str, Any]) -> bytes:
    """Generate an image with Qwen-Image through Alibaba Cloud Model Studio (DashScope SDK).

    The API returns a temporary URL on Alibaba's storage, which is downloaded straight away.
    """
    import dashscope
    from dashscope import MultiModalConversation

    dashscope.base_http_api_url = DASHSCOPE_BASE_URL
    response = MultiModalConversation.call(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        result_format="message",
        stream=False,
        **params,
    )
    if response.status_code != 200:
        message = f"{response.code}: {response.message}"
        if "DataInspection" in str(response.code):  # Alibaba's content-moderation error code
            raise GenerationRefused(message)
        raise RuntimeError(message)
    image_url = response.output.choices[0].message.content[0]["image"]
    return download_image(image_url, allowed_host_suffix=".aliyuncs.com")


def download_image(url: str, allowed_host_suffix: str, timeout_seconds: int = 60) -> bytes:
    """Download an image, refusing URLs that aren't HTTPS on the provider's own storage."""
    parsed = urlparse(url)
    if parsed.scheme != "https" or not (parsed.hostname or "").endswith(allowed_host_suffix):
        raise ValueError(f"Unexpected image URL: {url}")
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return response.read()


# --- Generators --------------------------------------------------------------------------------

GENERATORS: dict[str, Generator] = {
    "gpt_image_2": Generator(
        name="gpt_image_2",
        model_id="gpt-image-2-2026-04-21",  # dated snapshot of gpt-image-2
        params={"size": "1024x1024", "quality": "medium", "output_format": "png"},
        generate=generate_openai,
    ),
    "nano_banana_2_1": Generator(
        name="nano_banana_2_1",
        model_id="gemini-nano-banana-2.1",
        params={"aspect_ratio": "1:1"},  # 1:1 at the default 1K resolution is 1024x1024
        generate=generate_gemini,
    ),
    "qwen_image_2_0": Generator(
        name="qwen_image_2_0",
        model_id="qwen-image-2.0-2026-03-03",  # dated snapshot of qwen-image-2.0
        params={
            "size": "1024*1024",
            "prompt_extend": False,  # don't let the API rewrite the prompt
            "watermark": False,  # otherwise a visible "Qwen-Image" mark is added
        },
        generate=generate_qwen,
    ),
}


# --- Running -----------------------------------------------------------------------------------


def save_png(image_bytes: bytes, path: Path) -> tuple[int, int]:
    """Decode the image and re-save it as PNG, so every generator's output has the same format.

    Returns:
        The image's (width, height).
    """
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        image.save(path, format="PNG")
        return image.size


def call_with_retries(
    generator: Generator, prompt: str, max_attempts: int, base_delay_seconds: float
) -> bytes:
    """Call a generator, retrying with exponential backoff on errors other than refusals."""
    for attempt in range(1, max_attempts + 1):
        try:
            return generator.generate(prompt, generator.model_id, generator.params)
        except GenerationRefused:
            raise
        except Exception as error:
            if attempt == max_attempts:
                raise
            delay = base_delay_seconds * 2 ** (attempt - 1)
            logger.warning(
                f"{generator.name} attempt {attempt} failed ({error!r}); retrying in {delay}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read every row of the manifest, or an empty list if it doesn't exist yet."""
    if not manifest_path.exists():
        return []
    with manifest_path.open(newline="") as file:
        return list(csv.DictReader(file))


def append_manifest_row(manifest_path: Path, row: dict[str, Any]) -> None:
    """Append one row to the manifest, writing the header if the file is new."""
    is_new = not manifest_path.exists()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def generate_images(
    prompts: list[PromptRecord],
    generators: list[Generator],
    output_root: Path,
    manifest_path: Path,
    delay_seconds: float = 1.0,
    max_attempts: int = 3,
) -> None:
    """Generate one image per prompt per generator, logging every attempt to the manifest.

    Prompts that already succeeded, or were refused, for a generator are skipped. Prompts that
    failed with an error are tried again on the next run.

    Args:
        prompts: prompts to generate from, e.g. from `load_prompts`.
        generators: generators to use, e.g. `[GENERATORS["gpt_image_2"]]`.
        output_root: images go in `output_root / generator.name`.
        manifest_path: CSV file logging every attempt.
        delay_seconds: pause between requests, to stay under rate limits.
        max_attempts: attempts per image before recording an error.
    """
    finished = {
        (row["prompt_id"], row["generator"])
        for row in read_manifest(manifest_path)
        if row["status"] in {"ok", "refused"}
    }

    for generator in generators:
        output_dir = output_root / generator.name
        output_dir.mkdir(parents=True, exist_ok=True)

        for record in tqdm(prompts, desc=generator.name):
            if (record.prompt_id, generator.name) in finished:
                continue

            row: dict[str, Any] = {
                "prompt_id": record.prompt_id,
                "generator": generator.name,
                "model_id": generator.model_id,
                "params": json.dumps(generator.params),
                "prompt": record.prompt,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            try:
                image_bytes = call_with_retries(generator, record.prompt, max_attempts, 5.0)
                filename = f"{generator.name}_{record.prompt_id}.png"
                width, height = save_png(image_bytes, output_dir / filename)
                row.update(status="ok", filename=filename, width=width, height=height)
            except GenerationRefused as error:
                row.update(status="refused", error=str(error)[:500])
            except Exception as error:
                logger.error(f"{generator.name} failed on {record.prompt_id}: {error!r}")
                row.update(status="error", error=repr(error)[:500])

            append_manifest_row(manifest_path, row)
            time.sleep(delay_seconds)


def prompts_completed_by_all(manifest_path: Path, generator_names: list[str]) -> list[str]:
    """IDs of prompts that every given generator produced an image for, in manifest order.

    Use this to pick the final test set (e.g. the first 300), so that every generator is
    tested on exactly the same prompts.
    """
    succeeded: dict[str, set[str]] = {}
    for row in read_manifest(manifest_path):
        if row["status"] == "ok":
            succeeded.setdefault(row["prompt_id"], set()).add(row["generator"])
    return [prompt_id for prompt_id, names in succeeded.items() if set(generator_names) <= names]


def main() -> None:
    """Command-line entry point; run with --help for options."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prompts", type=Path, default=Path("data/test_prompts/prompts.csv"))
    parser.add_argument("--generators", nargs="+", choices=list(GENERATORS), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/test/mllm"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/test_prompts/generation_manifest.csv")
    )
    parser.add_argument("--limit", type=int, help="only use the first N prompts (for a trial)")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    prompts = load_prompts(args.prompts)[: args.limit]
    generate_images(
        prompts,
        [GENERATORS[name] for name in args.generators],
        args.output_root,
        args.manifest,
        delay_seconds=args.delay,
    )


if __name__ == "__main__":
    main()
