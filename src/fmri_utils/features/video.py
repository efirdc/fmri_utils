"""Video features for encoding models: clip embeddings, cached, then put on a scanner clock.

The same two halves as the text features, for the same reason. A video model
sees a short clip, not a frame, so the unit here is a clip, and each clip is
stamped with the time of its centre. Embedding every clip of a stimulus is the
expensive step and happens once, into a cache. Placing those clip vectors on a
run's TR clock is cheap and is done per run, per participant, from the cache.

    from fmri_utils.features import video

    spec = video.VideoModelSpec()                  # VideoMAE-base, 16 frames at 7.5 Hz
    result = video.embed_video("clip.mp4", spec, duration=8.0)
    video.write("features/", spec, "clip", result, source="clip.mp4")

    entry = video.read("features/", spec, "clip")
    events = [video.Event(onset=12.0, duration=8.0, times=entry["times"],
                          values=entry["embeddings"][:, 11])]   # block 12
    grid, dense, on = video.timeline(events, tr_times)
    features = video.to_samples(dense, grid, tr_times)          # (TRs, 768)

**Why the clip series is zero-filled before it is resampled.** An experiment
with gaps between videos is not a continuous stimulus, and a normalised kernel
evaluated on the clip samples alone would treat a TR next to a video as if it
were inside it: the weights of the few clips in reach are renormalised to one,
so the TR gets full-amplitude video features. ``timeline`` instead writes the
clips onto a dense regular grid that is zero wherever no video is on, and
``to_samples`` low-passes that grid onto the TR times. On a dense regular grid
the kernel's weight sum is the same at every TR, so normalising only fixes its
gain, and a TR that is half inside a video gets half its features. The default
kernel is three-lobe Lanczos: a windowed sinc with its cutoff at the TR's
Nyquist frequency, which is the right anti-aliasing filter for decimating a
4 Hz clip series to a 0.5 Hz scanner clock, and the resampler the text
encoding literature and the VideoMAE encoding work this follows both use. Its
negative lobes ring slightly at a video's onset and offset; ``kernel="hann"``
trades that for more blur.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from . import resample as _resample

SCHEMA = "fmri_utils.features.video/1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VideoModelSpec:
    """A video model and how its clips are cut.

    ``frame_rate`` is the rate of the frames the model is shown, not of the
    source file. VideoMAE was trained on 16 frames taken every fourth frame of
    30 fps Kinetics video, i.e. 7.5 Hz over 2.13 s, so that is the default.
    Source frames are picked by nearest time, so 25, 30 and 60 fps files all
    give the model the same temporal spacing.

    ``hop`` is the spacing of clip centres in seconds; 0.25 s (4 Hz) is well
    above what a 2 s TR can resolve. ``layers`` are 1-based transformer blocks
    to keep, each mean-pooled over its tokens; ``None`` keeps them all, which
    costs almost nothing beyond the model pass itself.
    """

    id: str = "videomae_base"
    huggingface_id: str = "MCG-NJU/videomae-base"
    frames_per_clip: int = 16
    frame_rate: float = 7.5
    hop: float = 0.25
    image_size: int = 224
    layers: Optional[tuple[int, ...]] = None
    batch_size: int = 4
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frames_per_clip < 1 or self.frame_rate <= 0 or self.hop <= 0:
            raise ValueError(f"{self.id}: frames_per_clip, frame_rate and hop must be positive")
        if self.layers is not None and min(self.layers) < 1:
            raise ValueError(f"{self.id}: layers are 1-based")

    @property
    def clip_seconds(self) -> float:
        return self.frames_per_clip / self.frame_rate


def clip_centres(duration: float, spec: VideoModelSpec) -> np.ndarray:
    """Clip-centre times covering ``[0, duration)`` at the spec's hop.

    Centres are at half-hop offsets, so each clip stands for the hop interval
    around it and the series tiles the video without favouring either end.
    """
    n = int(np.floor(duration / spec.hop + 1e-9))
    return (np.arange(n) + 0.5) * spec.hop


def clip_frame_times(centre: float, duration: float, spec: VideoModelSpec) -> np.ndarray:
    """Times of the frames one clip is built from, clamped to the video.

    A clip near the start or end would reach outside the video; its frames are
    clamped to the first or last, so edge clips see a held frame rather than
    being dropped. The first and last clip therefore carry a little less
    motion than the rest.
    """
    offsets = (np.arange(spec.frames_per_clip) - (spec.frames_per_clip - 1) / 2.0) / spec.frame_rate
    return np.clip(centre + offsets, 0.0, max(duration - 1e-6, 0.0))


def read_frames(path, start: float = 0.0, duration: Optional[float] = None,
                size: int = 224) -> tuple[np.ndarray, float]:
    """Decode a video segment, resized and centre-cropped for the model.

    Every source frame in ``[start, start + duration)`` is resized so its short
    side is ``size`` and centre-cropped to a square, the preprocessing VideoMAE
    was trained with. Returns ``(frames, fps)`` with frames as ``(n, size,
    size, 3)`` RGB uint8.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        raise ValueError(f"{path}: no frame rate")
    first = int(round(start * fps))
    last = None if duration is None else first + int(round(duration * fps))
    if first:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    index = first
    while last is None or index < last:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        scale = size / min(height, width)
        resized = cv2.resize(frame, (max(size, round(width * scale)), max(size, round(height * scale))),
                             interpolation=cv2.INTER_AREA)
        top = (resized.shape[0] - size) // 2
        left = (resized.shape[1] - size) // 2
        frames.append(cv2.cvtColor(resized[top:top + size, left:left + size], cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"{path}: no frames decoded from {start} s")
    return np.stack(frames), fps


def load_videomae(spec: VideoModelSpec, device: str):
    from transformers import VideoMAEModel

    model, info = VideoMAEModel.from_pretrained(spec.huggingface_id, output_loading_info=True)
    # Some transformers releases (5.17 among them) fail to map the checkpoint's
    # q_bias/v_bias onto the attention layers and initialise them at random,
    # reporting it only as a warning. Features from such a model are not
    # VideoMAE's, so refuse it.
    missing = [key for key in info.get("missing_keys", []) if "encoder" in key or "embeddings" in key]
    if missing:
        raise RuntimeError(f"{spec.huggingface_id}: weights missing from the checkpoint "
                           f"(transformers version?): {sorted(missing)[:4]}")
    if getattr(model.config, "num_frames", spec.frames_per_clip) != spec.frames_per_clip:
        raise ValueError(f"{spec.huggingface_id} takes {model.config.num_frames} frames, "
                         f"spec asks for {spec.frames_per_clip}")
    return model.to(device).eval()


def embed_video(path, spec: VideoModelSpec = VideoModelSpec(), device: str = "",
                start: float = 0.0, duration: Optional[float] = None, model=None,
                progress: Callable[[int, int], None] | None = None) -> dict:
    """Embed every clip of one video segment.

    Returns a dict with ``embeddings`` as ``(clips, layers, dims)`` float32,
    ``times`` (clip centres in seconds from ``start``), ``layers`` (the 1-based
    block of each layer row), and the frame indices each clip was built from.
    Pass ``model`` to reuse one loaded model across videos.
    """
    import torch

    from .extract import select_device

    device = select_device(device)
    frames, fps = read_frames(path, start=start, duration=duration, size=spec.image_size)
    length = frames.shape[0] / fps if duration is None else min(duration, frames.shape[0] / fps)
    times = clip_centres(length, spec)
    indices = np.stack([
        np.clip(np.round(clip_frame_times(t, length, spec) * fps).astype(int), 0, frames.shape[0] - 1)
        for t in times
    ])
    model = model if model is not None else load_videomae(spec, device)
    n_blocks = int(model.config.num_hidden_layers)
    layers = spec.layers or tuple(range(1, n_blocks + 1))
    if max(layers) > n_blocks:
        raise ValueError(f"asked for block {max(layers)}, model has {n_blocks}")

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pixels = (torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0 - mean) / std
    out = []
    with torch.inference_mode():
        for begin in range(0, len(times), spec.batch_size):
            batch = torch.from_numpy(indices[begin:begin + spec.batch_size])
            clips = pixels[batch].to(device)                      # (b, frames, 3, H, W)
            states = model(pixel_values=clips, output_hidden_states=True).hidden_states
            pooled = torch.stack([states[layer].mean(dim=1) for layer in layers], dim=1)
            out.append(pooled.float().cpu().numpy())
            if progress:
                progress(min(begin + spec.batch_size, len(times)), len(times))
    return {
        "embeddings": np.concatenate(out).astype(np.float32),
        "times": times,
        "layers": np.asarray(layers, dtype=int),
        "frame_indices": indices,
        "source_fps": fps,
        "source_frames": int(frames.shape[0]),
        "start": float(start),
        "duration": float(length),
    }


# Cache ---------------------------------------------------------------------

def _file_digest(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def fingerprint(spec: VideoModelSpec, source, start: float = 0.0,
                duration: Optional[float] = None) -> str:
    """Identify an entry by the model, the clip cut, the segment and the file's bytes."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"spec": asdict(spec), "start": start, "duration": duration},
                             sort_keys=True, default=str).encode("utf-8"))
    digest.update(_file_digest(source).encode("ascii"))
    return digest.hexdigest()[:16]


def entry_path(root, spec: VideoModelSpec, stimulus: str) -> Path:
    return Path(root) / spec.id / f"{stimulus}.npz"


def write(root, spec: VideoModelSpec, stimulus: str, result: dict, source,
          start: float = 0.0, duration: Optional[float] = None, extra: dict | None = None) -> Path:
    """Store one stimulus worth of clip embeddings and their times."""
    target = entry_path(root, spec, stimulus)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        embeddings=np.asarray(result["embeddings"], dtype=np.float32),
        times=np.asarray(result["times"], dtype=np.float64),
        layers=np.asarray(result["layers"], dtype=int),
        frame_indices=np.asarray(result["frame_indices"], dtype=int),
        meta=json.dumps({
            "schema": SCHEMA,
            "feature_id": spec.id,
            "spec": asdict(spec),
            "stimulus": stimulus,
            "source": str(source),
            "source_fps": result["source_fps"],
            "source_frames": result["source_frames"],
            "start": result["start"],
            "duration": result["duration"],
            "n_clips": int(len(result["times"])),
            "dimensions": int(result["embeddings"].shape[-1]),
            "fingerprint": fingerprint(spec, source, start, duration),
            **(extra or {}),
        }),
    )
    return target


def read(root, spec: VideoModelSpec, stimulus: str) -> dict:
    with np.load(entry_path(root, spec, stimulus), allow_pickle=False) as handle:
        return {
            "embeddings": handle["embeddings"],
            "times": handle["times"],
            "layers": handle["layers"],
            "frame_indices": handle["frame_indices"],
            "meta": json.loads(str(handle["meta"])),
        }


def is_current(root, spec: VideoModelSpec, stimulus: str, source, start: float = 0.0,
               duration: Optional[float] = None) -> bool:
    path = entry_path(root, spec, stimulus)
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as handle:
            meta = json.loads(str(handle["meta"]))
    except Exception:
        return False
    return meta.get("fingerprint") == fingerprint(spec, source, start, duration)


# Scanner clock -------------------------------------------------------------

@dataclass(frozen=True)
class Event:
    """One presentation of a video on a run's clock.

    ``onset`` and ``duration`` are seconds on the same clock as the sample
    times. ``times`` are the clip centres relative to the video's first frame
    and ``values`` the clip vectors, ``(clips, dims)``.
    """

    onset: float
    duration: float
    times: np.ndarray
    values: np.ndarray


def timeline(events: Sequence[Event], sample_times, step: float = 0.125,
             pad: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Write events onto a dense regular grid that is zero between videos.

    Inside a video the clip series is linearly interpolated to the grid, held
    at the first and last clip up to the video's edges. Returns ``(grid,
    values, on)``: grid times, ``(grid, dims)`` values, and a 0/1 video-on
    indicator. The grid extends ``pad`` seconds beyond the samples on both
    sides so the resampling kernel never runs off its end.
    """
    sample_times = np.asarray(sample_times, dtype=np.float64)
    grid = np.arange(sample_times[0] - pad, sample_times[-1] + pad + step / 2, step)
    dims = {np.asarray(event.values).shape[1] for event in events}
    if len(dims) != 1:
        raise ValueError(f"events disagree on dimensions: {sorted(dims)}")
    values = np.zeros((grid.size, dims.pop()), dtype=np.float64)
    on = np.zeros(grid.size, dtype=np.float64)
    for event in events:
        inside = (grid >= event.onset) & (grid < event.onset + event.duration)
        if np.any(on[inside]):
            raise ValueError(f"event at {event.onset} s overlaps another")
        local = grid[inside] - event.onset
        event_values = np.asarray(event.values, dtype=np.float64)
        values[inside] = np.stack([np.interp(local, event.times, column)
                                   for column in event_values.T], axis=1)
        on[inside] = 1.0
    return grid, values, on


def to_samples(values, grid, sample_times, kernel: str = "lanczos", **kernel_args) -> np.ndarray:
    """Low-pass a dense timeline onto the sample times; see the module docstring."""
    return _resample.resample(values, grid, sample_times, kernel=kernel, normalise=True,
                              **kernel_args)
