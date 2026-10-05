import numpy as np
import pytest

from fmri_utils.features import video


def _event(onset, value=1.0, dims=1, duration=8.0):
    spec = video.VideoModelSpec()
    times = video.clip_centres(duration, spec)
    return video.Event(onset=onset, duration=duration, times=times,
                       values=np.full((times.size, dims), value))


def test_clip_centres_tile_the_video():
    spec = video.VideoModelSpec(hop=0.25)
    times = video.clip_centres(8.0, spec)
    assert times.size == 32
    assert times[0] == pytest.approx(0.125) and times[-1] == pytest.approx(7.875)


def test_clip_frames_are_clamped_at_the_edges():
    spec = video.VideoModelSpec()
    first = video.clip_frame_times(0.125, 8.0, spec)
    assert first.size == 16 and first.min() == 0.0
    assert np.all(np.diff(first) >= 0)
    assert np.allclose(np.diff(video.clip_frame_times(4.0, 8.0, spec)), 1 / 7.5)


def test_zero_filled_timeline_scales_partial_coverage():
    samples = np.arange(30) * 2.0 + 1.0
    grid, dense, on = video.timeline([_event(20.0), _event(46.0)], samples)
    features = video.to_samples(dense, grid, samples)[:, 0]
    # Well inside a video: full amplitude. Far from any video: zero.
    assert features[np.searchsorted(samples, 23.0)] == pytest.approx(1.0, abs=0.05)
    assert abs(features[np.searchsorted(samples, 37.0)]) < 0.05
    # A TR centred on the video's onset is about half inside it.
    grid2, dense2, _ = video.timeline([_event(21.0)], samples)
    at_onset = video.to_samples(dense2, grid2, samples)[np.searchsorted(samples, 21.0), 0]
    assert at_onset == pytest.approx(0.5, abs=0.1)
    assert on.max() == 1.0 and on.min() == 0.0


def test_timeline_follows_the_clip_series():
    spec = video.VideoModelSpec()
    times = video.clip_centres(8.0, spec)
    ramp = video.Event(onset=10.0, duration=8.0, times=times, values=times[:, None])
    samples = np.arange(20) * 0.5
    grid, dense, _ = video.timeline([ramp], samples, step=0.125)
    inside = (grid >= 10.125) & (grid <= 17.875)
    assert np.allclose(dense[inside, 0], grid[inside] - 10.0)


def test_overlapping_events_are_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        video.timeline([_event(10.0), _event(15.0)], np.arange(20) * 2.0)
