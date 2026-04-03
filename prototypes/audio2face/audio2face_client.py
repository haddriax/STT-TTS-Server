"""
Audio2Face-3D gRPC client.

Connects to a locally running Nvidia Audio2Face-3D microservice
(default: localhost:52000) and converts synthesised audio into ARKit
blendshape frames.

Setup
-----
See audio2face_install.md for full instructions.

gRPC API — actual module layout (nvidia_ace-1.2.0)
---------------------------------------------------
- Stub    : nvidia_ace.services.a2f_controller.v1_pb2_grpc.A2FControllerServiceStub
- Input   : nvidia_ace.controller.v1_pb2.AudioStream  (oneof: audio_stream_header | audio_with_emotion | end_of_audio)
            AudioStreamHeader.audio_header → nvidia_ace.audio.v1_pb2.AudioHeader
            audio_with_emotion             → nvidia_ace.a2f.v1_pb2.AudioWithEmotion (audio_buffer: bytes)
- Output  : nvidia_ace.controller.v1_pb2.AnimationDataStream
            .animation_data_stream_header.skel_animation_header.blend_shapes  → list[str]  (names, sent once)
            .animation_data.skel_animation.blend_shape_weights                → FloatArrayWithTimeCode
              .time_code  (float)
              .values     (repeated float)
- Audio   : PCM 16-bit, 16 kHz, mono
- FPS     : 30 (fixed by service)
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
from scipy.signal import resample_poly

from prototypes.audio2face.config import Audio2FaceConfig
from tts import BlendShapeFrame

logger = logging.getLogger(__name__)

_A2F_SAMPLE_RATE = 16_000  # Hz — required by A2F-3D
_A2F_FPS = 30              # A2F always outputs at 30 FPS

try:
    import grpc  # noqa: F401
    from nvidia_ace.services.a2f_controller import (  # type: ignore[import]
        v1_pb2_grpc as _a2f_grpc,
    )
    from nvidia_ace.controller import v1_pb2 as _ctrl  # type: ignore[import]
    from nvidia_ace.audio import v1_pb2 as _audio      # type: ignore[import]
    from nvidia_ace.a2f import v1_pb2 as _a2f          # type: ignore[import]
    _GRPC_AVAILABLE = True
except ImportError:
    _GRPC_AVAILABLE = False


class Audio2FaceError(Exception):
    """Raised when the Audio2Face-3D gRPC call fails."""


class Audio2FaceClient:
    """Thin async wrapper around the Audio2Face-3D gRPC service."""

    def __init__(self, cfg: Audio2FaceConfig) -> None:
        self.cfg = cfg
        self._endpoint = f"{cfg.host}:{cfg.port}"

    async def generate_blendshapes(
        self,
        audio_f32: np.ndarray,
        source_sample_rate: int,
    ) -> list[BlendShapeFrame]:
        """Convert float32 audio to ARKit blendshape frames via Audio2Face-3D.

        Parameters
        ----------
        audio_f32:
            Mono float32 audio array at *source_sample_rate*.
        source_sample_rate:
            Sample rate of *audio_f32* (e.g. 24000 for Kokoro native).

        Returns
        -------
        list[BlendShapeFrame]
            One frame per 1/30 s of audio, each with ~52 ARKit blendshape weights.

        Raises
        ------
        Audio2FaceError
            On import failure (nvidia_ace not installed) or any gRPC error.
        """
        if not _GRPC_AVAILABLE:
            raise Audio2FaceError(
                "nvidia_ace gRPC stubs not installed. "
                "Install the nvidia_ace wheel from NVIDIA/Audio2Face-3D-Samples "
                "and grpcio, then restart the server."
            )

        pcm_int16 = _resample_to_a2f(audio_f32, source_sample_rate)

        try:
            frames = await asyncio.to_thread(self._call_grpc, pcm_int16)
        except Audio2FaceError:
            raise
        except Exception as exc:
            raise Audio2FaceError(f"Audio2Face-3D gRPC error: {exc}") from exc

        return frames

    def _call_grpc(self, pcm_int16: bytes) -> list[BlendShapeFrame]:
        """Blocking gRPC call — runs in a thread via asyncio.to_thread."""
        import grpc  # noqa: WPS433
        from nvidia_ace.services.a2f_controller import v1_pb2_grpc as a2f_grpc  # type: ignore[import]
        from nvidia_ace.controller import v1_pb2 as ctrl                         # type: ignore[import]
        from nvidia_ace.audio import v1_pb2 as audio                             # type: ignore[import]
        from nvidia_ace.a2f import v1_pb2 as a2f                                 # type: ignore[import]

        channel = grpc.insecure_channel(self._endpoint)
        stub = a2f_grpc.A2FControllerServiceStub(channel)

        def _request_iterator():
            # First message: stream header with audio format
            yield ctrl.AudioStream(
                audio_stream_header=ctrl.AudioStreamHeader(
                    audio_header=audio.AudioHeader(
                        samples_per_second=_A2F_SAMPLE_RATE,
                        bits_per_sample=16,
                        channel_count=1,
                        audio_format=audio.AudioHeader.AUDIO_FORMAT_PCM,
                    )
                )
            )
            # Second message: raw PCM bytes
            yield ctrl.AudioStream(
                audio_with_emotion=a2f.AudioWithEmotion(
                    audio_buffer=pcm_int16,
                )
            )
            # Third message: signal end of audio
            yield ctrl.AudioStream(
                end_of_audio=ctrl.AudioStream.EndOfAudio()
            )

        frames: list[BlendShapeFrame] = []
        blend_shape_names: list[str] = []
        try:
            responses = stub.ProcessAudioStream(
                _request_iterator(),
                timeout=self.cfg.timeout,
            )
            for response in responses:
                # The header response carries blend shape names; data responses carry weights
                if response.HasField("animation_data_stream_header"):
                    blend_shape_names = list(
                        response.animation_data_stream_header.skel_animation_header.blend_shapes
                    )
                if response.HasField("animation_data"):
                    bsw = response.animation_data.skel_animation.blend_shape_weights
                    blendshapes = {
                        name: float(weight)
                        for name, weight in zip(blend_shape_names, bsw.values)
                    }
                    frames.append(BlendShapeFrame(time=bsw.time_code, blendshapes=blendshapes))
        except grpc.RpcError as exc:
            raise Audio2FaceError(
                f"Audio2Face-3D gRPC call failed ({exc.code()}): {exc.details()}"
            ) from exc
        finally:
            channel.close()

        logger.info("audio2face: received %d blendshape frames @ %d FPS", len(frames), _A2F_FPS)
        return frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resample_to_a2f(audio_f32: np.ndarray, source_rate: int) -> bytes:
    """Resample float32 mono audio to 16 kHz PCM int16 bytes."""
    if source_rate != _A2F_SAMPLE_RATE:
        from math import gcd
        g = gcd(source_rate, _A2F_SAMPLE_RATE)
        audio_f32 = resample_poly(audio_f32, _A2F_SAMPLE_RATE // g, source_rate // g)
    pcm = np.clip(audio_f32, -1.0, 1.0)
    return (pcm * 32767).astype(np.int16).tobytes()
