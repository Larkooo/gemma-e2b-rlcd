"""Frozen Gemma baseline: one forward pass per question, no text generation."""

import json
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from string import ascii_uppercase
from tempfile import TemporaryDirectory

from .core import State, TokenScores


def media_info(path: str) -> dict:
    if shutil.which("ffprobe") is None:
        raise RuntimeError("Install ffmpeg (including ffprobe) to validate audio/video durations")
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@contextmanager
def audio_paths(state: State) -> Iterator[list[str]]:
    """Include a video's soundtrack instead of silently treating it as silent video."""
    import math

    def duration(info: dict, limit: float, kind: str) -> None:
        seconds = float(info.get("format", {}).get("duration", "nan"))
        if not math.isfinite(seconds) or seconds <= 0 or seconds > limit:
            raise ValueError(f"{kind} duration must be known, positive, and at most {limit}s")

    paths = list(state.audio)
    for path in paths:
        duration(media_info(path), 30, "Audio")
    if not state.videos:
        yield paths
        return
    video = state.videos[0]
    info = media_info(video)
    duration(info, 60, "Video")
    has_audio = any(stream.get("codec_type") == "audio" for stream in info["streams"])
    if not has_audio:
        yield paths
        return
    if paths:
        raise ValueError(
            "A video soundtrack and a separate audio clip require two audio streams; supply one"
        )
    duration(info, 30, "Video with soundtrack")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to include the video soundtrack")
    with TemporaryDirectory(prefix="gemma-video-") as directory:
        extracted = str(Path(directory) / "soundtrack.wav")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", extracted],
            capture_output=True,
            check=True,
        )
        yield [extracted]


class MLXBackend:
    def __init__(self, model: str, *, max_input_tokens: int = 8192):
        import mlx.core as mx
        from mlx_vlm import load

        self.mx = mx
        self.model_id = model
        self.model, self.processor = load(model, strict=True)
        if self.model.config.model_type != "gemma4":
            raise ValueError("This backend currently targets Gemma 4")
        if self.model.audio_tower is None:
            raise ValueError("Use the full E2B checkpoint with its audio encoder")
        from .vision import install_indexed_positions

        install_indexed_positions(self.model)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.max_input_tokens = max_input_tokens
        self._symbols: list[str] = []
        self._ids: dict[str, int] = {}
        for width in (1, 2, 3):
            for letters in product(ascii_uppercase, repeat=width):
                symbol = "".join(letters)
                ids = self.tokenizer.encode(symbol, add_special_tokens=False)
                if len(ids) == 1 and self.tokenizer.decode(ids) == symbol:
                    self._symbols.append(symbol)
                    self._ids[symbol] = ids[0]
                if len(self._symbols) == 255:
                    break
            if len(self._symbols) == 255:
                break
        if len(self._symbols) < 255:
            raise ValueError("Tokenizer does not provide 255 distinct single-token option codes")

    def symbols(self, count: int) -> Sequence[str]:
        if not 2 <= count <= 255:
            raise ValueError("Expected 2 to 255 choices")
        return self._symbols[:count]

    def score(self, state: State, prompt: str, symbols: Sequence[str]) -> TokenScores:
        from mlx_vlm.utils import prepare_inputs

        mx = self.mx
        with audio_paths(state) as audios:
            content = [{"type": "image"} for _ in state.images]
            content += [{"type": "video", "video": path} for path in state.videos]
            content.append({"type": "text", "text": json.dumps({"state": state.text})})
            content += [{"type": "audio"} for _ in audios]
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ]
            formatted = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = dict(
                prepare_inputs(
                    self.processor,
                    prompts=formatted,
                    images=list(state.images) or None,
                    audio=audios or None,
                    videos=list(state.videos) or None,
                    fps=1.0,
                    max_frames=60,
                )
            )
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > self.max_input_tokens:
            raise ValueError(
                f"Input has {input_tokens} tokens; limit is {self.max_input_tokens}. No truncation applied."
            )
        # Each request has one unpadded sequence. Do not pass a 2D tokenizer mask
        # as the decoder's causal attention mask.
        inputs.pop("attention_mask", None)
        output = self.model(**inputs, logits_to_keep=1)
        logits = output.logits[0, -1, :].astype(mx.float32)
        ids = mx.array([self._ids[symbol] for symbol in symbols])
        selected = logits[ids]
        mass = mx.exp(mx.logsumexp(selected) - mx.logsumexp(logits))
        mx.eval(selected, mass)
        return TokenScores(tuple(selected.tolist()), min(1.0, float(mass.item())), input_tokens)
