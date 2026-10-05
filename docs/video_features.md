# Video Features

`fmri_utils.features.video` does for videos what [Text Features](features.md)
does for transcripts:

1. **Embed** a video's clips with a video model, once, into a cache.
2. **Place** the cached clip series on a run's scanner clock, for every
   presentation of the video, as often as you like and with no model loaded.

The unit is a clip, not a frame: a video transformer sees a short stack of
frames, so each vector is stamped with the time of its clip's centre.

## Quick Start

```python
import numpy as np
from fmri_utils.features import video

spec = video.VideoModelSpec()                 # VideoMAE-base, 16 frames at 7.5 Hz, hop 0.25 s
result = video.embed_video("clip.mp4", spec, duration=8.0)
video.write("features/", spec, "clip", result, source="clip.mp4", duration=8.0)

# later, per run
entry = video.read("features/", spec, "clip")
tr_times = 2.0 * np.arange(166) + 0.96        # volume times after slice-timing correction
events = [video.Event(onset=o, duration=8.0, times=entry["times"],
                      values=entry["embeddings"][:, 11])      # block 12 of 12
          for o in onsets]
grid, dense, on = video.timeline(events, tr_times)
features = video.to_samples(dense, grid, tr_times)            # (166, 768)
```

`embed_video` returns `embeddings` as `(clips, layers, dims)`: every block of
the model, each mean-pooled over its tokens. Keeping them all costs almost
nothing next to the model pass, and choosing a layer later needs no re-run.
`is_current` checks a cache entry against the model spec, the segment, and a
hash of the video file's bytes.

## Clips

`VideoModelSpec` is the model plus the clip cut:

- `frames_per_clip` and `frame_rate` are what the model is shown. VideoMAE was
  trained on 16 frames taken every fourth frame of 30 fps video (7.5 Hz, 2.13
  s), so those are the defaults. Source frames are chosen by nearest time, so
  a 60 fps file gives the model the same spacing as a 30 fps one.
- `hop` spaces the clip centres; 0.25 s tiles an 8 s video with 32 clips at
  0.125, 0.375, ... 7.875 s.
- Frames are resized to a 224 short side and centre-cropped, VideoMAE's own
  preprocessing. A clip that would reach past the start or end holds the edge
  frame, so the first and last clips carry a little less motion.

## Placing Clips On A Scanner Clock

Two things make video features different from a continuous transcript.
Experiments leave gaps between videos, and the clip series (4 Hz) is much
faster than a TR (0.5 Hz at 2 s).

`timeline` writes each presentation onto a dense regular grid (0.125 s by
default) that is **zero wherever no video is on**. Inside a video the clip
series is interpolated linearly. `to_samples` then low-passes that grid onto
the TR times, by default with a normalised three-lobe Lanczos kernel. That
kernel is a windowed sinc with its cutoff at the TR's Nyquist frequency, which
is the anti-aliasing filter a 4 Hz to 0.5 Hz decimation calls for. The
language-encoding literature uses it, and so does the VideoMAE encoding work
this module was written against.

The zero fill is what makes normalising safe. The alternative, a normalised
kernel over the clip samples alone, renormalises whichever few clips are in
reach, so a TR just after a video gets full-amplitude features. On a dense
regular grid the kernel's weight sum is the same at every TR, so normalising
only sets its gain: a TR half inside a video gets half its features, and a TR
between videos gets none. Lanczos rings slightly at a video's onset and
offset. `kernel="hann"` avoids that at the price of more temporal blur.

If the features should describe *what* is on screen rather than *that*
something is, centre each dimension on its mean over all clips of the
stimulus set before building the events. Give the on/off structure its own
regressor: `on` from `timeline`, resampled through the same kernel.

## Dependencies

Embedding needs `torch`, `transformers` and `opencv-python`. Reading the
cache, `timeline` and `to_samples` need only numpy.
