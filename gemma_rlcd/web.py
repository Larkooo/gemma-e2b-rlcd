"""Local decision playground with one resident model and serialized GPU requests."""

import argparse
import asyncio
import json
import logging
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .core import DecisionEngine, Independent, State, parse_question

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
STATIC = Path(__file__).parent / "static"
LOGGER = logging.getLogger(__name__)


def normalize_question(raw: dict) -> dict:
    question = dict(raw)
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind in {"choice", "independent"}:
        if isinstance(criteria, list):
            if not all(isinstance(name, str) and name.strip() for name in criteria):
                raise ValueError("Choice names must be nonempty strings")
            if len(set(criteria)) != len(criteria):
                raise ValueError("Choice names must be unique")
            criteria = dict.fromkeys(criteria, "")
        if isinstance(criteria, dict):
            question["criteria"] = {
                name: name
                if description is None
                or description == ""
                or isinstance(description, str)
                and not description.strip()
                else description
                for name, description in criteria.items()
            }
    elif kind == "score":
        if criteria is None:
            count = question.pop("levels", 5)
            if type(count) is not int or not 2 <= count <= 10:
                raise ValueError("Choose between 2 and 10 grade levels")
            criteria = [""] * count
        if isinstance(criteria, list):
            question["criteria"] = [
                f"Grade {index} on a scale from 0 (lowest) to {len(criteria) - 1} (highest)"
                if description is None or isinstance(description, str) and not description.strip()
                else str(description)
                if type(description) in (int, float)
                else description
                for index, description in enumerate(criteria)
            ]
    return question


def read_spec(encoded: str) -> tuple[dict, dict]:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    spec = json.loads(encoded, object_pairs_hook=unique_object)
    if not isinstance(spec, dict) or set(spec) - {"text", "instructions", "questions", "media"}:
        raise ValueError("Request accepts text, instructions, questions, and uploaded media")
    if not isinstance(spec.get("text", ""), str) or not isinstance(
        spec.get("instructions", ""), str
    ):
        raise ValueError("Text and shared instructions must be strings")
    raw_questions = spec.get("questions")
    if not isinstance(raw_questions, dict) or not 1 <= len(raw_questions) <= 32:
        raise ValueError("Define between 1 and 32 output fields")
    questions = {}
    for name, raw in raw_questions.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise ValueError("Each field needs a nonempty name of at most 100 characters")
        if not isinstance(raw, dict):
            raise ValueError(f"Field {name} must be a question object")
        raw = normalize_question(raw)
        spec["questions"][name] = raw
        question = parse_question(raw)
        if spec.get("instructions", "").strip():
            question = parse_question(
                {
                    **raw,
                    "instructions": f"{spec['instructions']}\n\n{question.instructions}",
                }
            )
        questions[name] = question
    field_count = sum(
        len(q.criteria) if isinstance(q, Independent) else 1 for q in questions.values()
    )
    if field_count > 128:
        raise ValueError("At most 128 individual fields or independent labels can run together")
    media = spec.get("media", [])
    if not isinstance(media, list) or len(media) > 10:
        raise ValueError("Attach at most 8 images, one audio clip, and one video")
    counts = {"image": 0, "audio": 0, "video": 0}
    for item in media:
        if not isinstance(item, dict) or set(item) != {"kind", "name"}:
            raise ValueError("Each attachment needs kind and name")
        if item["kind"] not in counts or not isinstance(item["name"], str):
            raise ValueError("Unknown media type")
        counts[item["kind"]] += 1
    if counts["image"] > 8 or counts["audio"] > 1 or counts["video"] > 1:
        raise ValueError("Attach at most 8 images, one audio clip, and one video")
    return spec, questions


async def read_upload(request: Request, directory: str):
    async with request.form(max_files=10, max_fields=1, max_part_size=1024 * 1024) as form:
        if set(form) - {"spec", "media"} or not isinstance(form.get("spec"), str):
            raise ValueError("Submit a JSON spec and optional media attachments")
        spec, questions = read_spec(form["spec"])
        files = form.getlist("media")
        if len(files) != len(spec.get("media", [])) or not all(
            isinstance(file, UploadFile) for file in files
        ):
            raise ValueError("Attachment files do not match the request")
        paths, total = [], 0
        for index, file in enumerate(files):
            suffix = Path(file.filename or "upload").suffix.lower()
            if len(suffix) > 10 or not suffix.replace(".", "").isalnum():
                suffix = ".bin"
            path = Path(directory) / f"attachment-{index}{suffix}"
            with path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError("Attachments exceed the 200 MB total upload limit")
                    output.write(chunk)
            paths.append(path)
        return spec, questions, paths


def inspect_media(path: Path) -> dict:
    try:
        process = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True,
            check=True,
            timeout=20,
        )
        return json.loads(process.stdout)
    except FileNotFoundError as exc:
        raise ValueError("Install ffmpeg and ffprobe to use audio or video") from exc
    except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ValueError(
            "Could not read this audio/video file. Try WAV, MP3, MP4, MOV, or WebM."
        ) from exc


def prepare_media(paths: list[Path], entries: list[dict]) -> tuple[dict, list[dict]]:
    state_paths = {"images": [], "audio": [], "videos": []}
    metadata = []
    for path, entry in zip(paths, entries, strict=True):
        kind = entry["kind"]
        item = {"name": entry["name"], "kind": kind, "bytes": path.stat().st_size}
        if kind == "image":
            try:
                with Image.open(path) as image:
                    frames = getattr(image, "n_frames", 1)
                    if frames > 1 and image.format not in {"JPEG", "MPO"}:
                        raise ValueError(
                            "This file is an animation. Use a video clip to evaluate its motion."
                        )
                    item.update(width=image.width, height=image.height)
                    if image.format in {"JPEG", "MPO"} and frames > 1:
                        # Phone HDR/depth JPEGs can contain multiple pictures;
                        # this does not make the primary photograph animated.
                        item.update(
                            source_format=image.format, embedded_images=frames, used_frame=0
                        )
                        image.seek(0)
                        photograph = ImageOps.exif_transpose(image).convert("RGB")
                        path = path.with_name(path.stem + "-primary.png")
                        photograph.save(path)
                    else:
                        image.verify()
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError("Could not read this image. Try PNG, JPEG, or WebP.") from exc
            state_paths["images"].append(str(path))
        else:
            info = inspect_media(path)
            types = {stream.get("codec_type") for stream in info.get("streams", [])}
            # Browsers often label an audio-only .webm upload as video/webm.
            # Use its actual streams so an exported recording still works.
            if kind == "video" and "video" not in types and "audio" in types:
                kind = "audio"
                item["kind"] = "audio"
            seconds = float(info.get("format", {}).get("duration", "nan"))
            limit = 30 if kind == "audio" or "audio" in types else 60
            if kind == "video" and (not math.isfinite(seconds) or not 0 < seconds <= limit):
                raise ValueError(f"Video must have a known duration of at most {limit} seconds")
            if kind == "audio" and math.isfinite(seconds) and not 0 < seconds <= 30:
                raise ValueError("Audio must be at most 30 seconds")
            if kind == "audio":
                if "audio" not in types or "video" in types:
                    raise ValueError(
                        "Attach video files as video, and audio-only files as speech/audio"
                    )
                converted = path.with_name(path.stem + "-audio.wav")
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-i",
                            str(path),
                            "-vn",
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            str(converted),
                        ],
                        capture_output=True,
                        check=True,
                        timeout=45,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise ValueError("Could not decode this audio clip") from exc
                # MediaRecorder WebM often has no duration metadata. Decode the
                # complete clip, then enforce the limit on the resulting WAV.
                seconds = float(inspect_media(converted).get("format", {}).get("duration", "nan"))
                if not math.isfinite(seconds) or not 0 < seconds <= 30:
                    raise ValueError(
                        "Audio must be at most 30 seconds; the recording was not trimmed"
                    )
                state_paths["audio"].append(str(converted))
            else:
                if "video" not in types:
                    raise ValueError("This attachment does not contain a video stream")
                state_paths["videos"].append(str(path))
            item.update(
                duration_seconds=seconds, has_soundtrack=kind == "video" and "audio" in types
            )
        metadata.append(item)
    if state_paths["audio"] and any(item.get("has_soundtrack") for item in metadata):
        raise ValueError(
            "The video already has a soundtrack. Remove the separate audio clip to use it."
        )
    return {key: tuple(values) for key, values in state_paths.items()}, metadata


class Runtime:
    def __init__(self, model: str, work_dir: Path, backend_factory=None):
        self.model = model
        self.work_dir = work_dir
        self.factory = backend_factory
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="decision-model")
        self.lock = asyncio.Lock()
        self.backend = None
        self.normal_backend = None
        self.error = None
        self.load_seconds = None

    async def load(self):
        def initialize():
            started = time.perf_counter()
            if self.factory is None:
                from .json_backend import JSONMLXBackend

                backend = JSONMLXBackend(self.model, branch_batch_size=8)
            else:
                backend = self.factory()
            from .comparison import generation_backend

            self.normal_backend = generation_backend(backend)
            self.backend = backend
            self.load_seconds = time.perf_counter() - started

        try:
            await asyncio.get_running_loop().run_in_executor(self.pool, initialize)
        except Exception:
            LOGGER.exception("Model initialization failed")
            self.error = "The model could not load. Check the model path and server log."

    def evaluate(
        self, spec: dict, questions: dict, paths: list[Path], comparison: bool = False, emit=None
    ) -> dict:
        started = time.perf_counter()
        media, metadata = prepare_media(paths, spec.get("media", []))
        normalized = time.perf_counter()
        state = State(text=spec.get("text", ""), **media)
        details = None
        if comparison:
            from .comparison import compare

            result, details = (
                compare(
                    self.backend,
                    state,
                    questions,
                    normalized - started,
                    emit=emit,
                    concurrent=True,
                    normal_backend=self.normal_backend,
                )
                if emit
                else compare(self.backend, state, questions, normalized - started)
            )
        else:
            result = DecisionEngine(self.backend).system_one(state, questions)
        finished = time.perf_counter()
        return {
            **result,
            "execution": result.get("execution", dict(self.backend.last_stats)),
            "media": metadata,
            "media_prepare_seconds": normalized - started,
            "decision_seconds": result.get("inference_seconds", finished - normalized),
            "total_seconds": finished - started,
            "model": "Gemma 4 E2B · frozen scorer",
            "load_seconds": self.load_seconds,
            **({"comparison": details} if details else {}),
        }


def create_app(model: str, work_dir: Path, backend_factory=None) -> FastAPI:
    runtime = Runtime(model, work_dir, backend_factory)

    @asynccontextmanager
    async def lifespan(app):
        work_dir.mkdir(parents=True, exist_ok=True)
        loading = asyncio.create_task(runtime.load())
        yield
        await loading
        runtime.pool.shutdown(wait=True)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.runtime = runtime
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        if request.method == "POST":
            origin = request.headers.get("origin")
            if origin and origin != f"http://{request.headers.get('host')}":
                return JSONResponse(
                    {"error": "Open the playground directly on localhost to run a request"},
                    status_code=403,
                )
            try:
                if int(request.headers.get("content-length", "0")) > MAX_UPLOAD_BYTES + 1024 * 1024:
                    return JSONResponse(
                        {"error": "Attachments exceed the 200 MB total upload limit"},
                        status_code=413,
                    )
            except ValueError:
                return JSONResponse({"error": "Invalid content length"}, status_code=400)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/demo")
    async def demo():
        return FileResponse(STATIC / "demo.html")

    @app.post("/api/compare-stream")
    async def stream(request: Request):
        if runtime.backend is None or runtime.error:
            return JSONResponse(
                {"error": runtime.error or "The model is still loading."}, status_code=503
            )
        if runtime.lock.locked():
            return JSONResponse(
                {"error": "Another evaluation is running. Wait for it to finish."}, status_code=429
            )

        await runtime.lock.acquire()
        try:
            directory = TemporaryDirectory(prefix="stream-", dir=work_dir)
        except BaseException:
            runtime.lock.release()
            raise
        try:
            spec, questions, paths = await read_upload(request, directory.name)
        except BaseException as exc:
            directory.cleanup()
            runtime.lock.release()
            if isinstance(exc, (ValueError, TypeError, KeyError)):
                return JSONResponse({"error": str(exc)}, status_code=400)
            raise

        async def events():
            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            stopped = Event()

            class StreamStopped(Exception):
                pass

            def emit(event):
                if stopped.is_set():
                    raise StreamStopped()
                loop.call_soon_threadsafe(queue.put_nowait, event)

            def evaluate():
                try:
                    result = runtime.evaluate(spec, questions, paths, True, emit)
                    emit({"type": "complete", "result": result})
                except StreamStopped:
                    pass
                except (ValueError, TypeError, KeyError) as exc:
                    loop.call_soon_threadsafe(
                        queue.put_nowait, {"type": "error", "error": str(exc)}
                    )
                except Exception:
                    LOGGER.exception("Streaming evaluation failed")
                    if not stopped.is_set():
                        loop.call_soon_threadsafe(
                            queue.put_nowait,
                            {
                                "type": "error",
                                "error": "Evaluation failed. Check the media or reduce the input size. Details are in the server log.",
                            },
                        )

            future = loop.run_in_executor(runtime.pool, evaluate)
            future.add_done_callback(lambda _: queue.put_nowait(None))
            try:
                yield json.dumps({"type": "accepted"}) + "\n"
                while (event := await queue.get()) is not None:
                    yield json.dumps(event, allow_nan=False) + "\n"
            finally:
                stopped.set()
                # Keep uploads and the GPU lock alive until the worker stops.
                with CancelScope(shield=True):
                    try:
                        await asyncio.shield(future)
                    finally:
                        directory.cleanup()
                        runtime.lock.release()

        return StreamingResponse(
            events(), media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no"}
        )

    @app.get("/api/status")
    async def status():
        return {
            "ready": runtime.backend is not None and runtime.error is None,
            "busy": runtime.lock.locked(),
            "error": runtime.error,
            "model": "Gemma 4 E2B",
            "batch_size": 8,
            "load_seconds": runtime.load_seconds,
        }

    @app.post("/api/compare")
    @app.post("/api/run")
    async def run(request: Request):
        if runtime.backend is None or runtime.error:
            return JSONResponse(
                {"error": runtime.error or "The model is still loading. Try again in a moment."},
                status_code=503,
            )
        if runtime.lock.locked():
            return JSONResponse(
                {"error": "Another evaluation is running. Wait for it to finish, then try again."},
                status_code=429,
            )
        async with runtime.lock:
            started = time.perf_counter()
            try:
                with TemporaryDirectory(prefix="run-", dir=work_dir) as directory:
                    spec, questions, paths = await read_upload(request, directory)
                    future = asyncio.get_running_loop().run_in_executor(
                        runtime.pool,
                        runtime.evaluate,
                        spec,
                        questions,
                        paths,
                        request.url.path == "/api/compare",
                    )
                    try:
                        result = await asyncio.shield(future)
                    except asyncio.CancelledError:
                        await future
                        raise
                result["request_seconds"] = time.perf_counter() - started
                return JSONResponse(result)
            except (ValueError, TypeError, KeyError) as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            except Exception:
                LOGGER.exception("Decision evaluation failed")
                return JSONResponse(
                    {
                        "error": "Evaluation failed. Check the media format or reduce the input size. Details are in the server log."
                    },
                    status_code=500,
                )

    @app.post("/api/validate")
    async def validate(request: Request):
        encoded = await request.body()
        if len(encoded) > 1024 * 1024:
            return JSONResponse({"error": "Schema exceeds 1 MB"}, status_code=413)
        try:
            spec, _ = read_spec(encoded.decode("utf-8"))
            return {"questions": spec["questions"]}
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Local multimodal decision playground")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--work-dir", type=Path, default=Path("work/web"))
    args = parser.parse_args()
    uvicorn.run(create_app(args.model, args.work_dir.resolve()), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
