import io
import itertools
from typing import BinaryIO, Union

import av
import numpy as np


def decode_audio(
    input_file: Union[str, BinaryIO],
    sampling_rate: int = 16000,
    split_stereo: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Decode the audio.

    Args:
        input_file: Path to the input file or a file-like object.
        sampling_rate: Resample the audio to this sample rate.
        split_stereo: Return separate left and right channels.

    Returns:
        A float32 Numpy array.

        If `split_stereo` is enabled, the function returns a 2-tuple with the
        separated left and right channels.
    """
    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono" if not split_stereo else "stereo",
        rate=sampling_rate,
    )

    with io.BytesIO() as raw_buffer:
        dtype = None

        with av.open(input_file, metadata_errors="ignore") as container:
            frames = container.decode(audio=0)
            frames = _ignore_invalid_frames(frames)
            frames = _group_frames(frames, 500000)
            frames = _resample_frames(frames, resampler)

            for frame in frames:
                array = frame.to_ndarray()
                dtype = array.dtype
                raw_buffer.write(array)

        audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

    # Convert s16 back to f32.
    audio = audio.astype(np.float32) / 32768.0

    if split_stereo:
        left_channel = audio[0::2]
        right_channel = audio[1::2]
        return left_channel, right_channel

    return audio


def _ignore_invalid_frames(
    frames: av.audio.frame.AudioFrame
) -> av.audio.frame.AudioFrame:
    """Ignore frames with invalid data errors."""
    iterator = iter(frames)

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except av.error.InvalidDataError:
            continue


def _group_frames(
    frames: av.audio.frame.AudioFrame, num_samples=None
) -> av.audio.frame.AudioFrame:
    """Group frames based on the number of samples."""
    fifo = av.audio.fifo.AudioFifo()

for frame in frames:
    frame.pts = None  # Ignore timestamp check.
    fifo.write(frame)

    if num_samples is not None and fifo.samples >= num_samples:
        yield fifo.read()

if fifo.samples > 0:
    yield fifo.read()
def _resample_frames( frames: av.audio.frame.AudioFrame, resampler: av.audio.resampler.AudioResampler ) -> av.audio.frame.AudioFrame: """Resample audio frames.""" # Add None to flush the resampler. for frame in itertools.chain(frames, [None]): yield from resampler.resample(frame)

all = ["decode_audio", "_ignore_invalid_frames", "_group_frames", "_resample_frames"]
