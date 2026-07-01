"""Shared utilities for inference: latent computation, noise, media I/O, video concat."""

from __future__ import annotations

import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
from torchvision.transforms import functional as TVF

from ltx_distillation.inference.memory_multishot import (
    audio_waveform_stats,
    normalize_audio_waveform_for_media,
)


def compute_latent_shapes(
    *,
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> tuple[list[int], list[int]]:
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(f"num_frames must be 1 + 8*k, got {num_frames}")

    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression

    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = float(audio_sample_rate) / float(audio_hop_length) / float(audio_latent_downsample)
    audio_frames = round(video_duration * audio_latent_fps)

    return (
        [batch_size, latent_frames, latent_channels, latent_h, latent_w],
        [batch_size, audio_frames, latent_channels],
    )


def add_noise(original: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    sigma = sigma.to(device=original.device, dtype=original.dtype)
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    return (1 - sigma) * original + sigma * noise


def frames_to_video_tensor(frames, target_h: int, target_w: int) -> torch.Tensor:
    tensors = []
    for idx, image in enumerate(frames):
        if image.size != (target_w, target_h):
            raise ValueError(
                f"Frame size mismatch at index {idx}: got={image.size}, expected={(target_w, target_h)}"
            )
        tensor = TVF.to_tensor(image)
        tensors.append(tensor * 2.0 - 1.0)
    return torch.stack(tensors, dim=1).contiguous()


@torch.no_grad()
def encode_memory_frames_batch(
    *,
    video_vae,
    batch_memory_frames,
    target_h: int,
    target_w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if getattr(video_vae, "encoder", None) is None:
        raise RuntimeError("video VAE encoder is not initialized for memory encoding")

    latents = []
    for memory_frames in batch_memory_frames:
        if not memory_frames:
            raise ValueError("memory_frames cannot be empty when encoding memory video")
        per_frame_latents = []
        for memory_item in memory_frames:
            is_clip_memory = isinstance(memory_item, list)
            frame_video = frames_to_video_tensor(
                memory_item if is_clip_memory else [memory_item],
                target_h,
                target_w,
            ).unsqueeze(0).to(device=device, dtype=dtype)
            latent = video_vae.encode(frame_video)
            del frame_video
            latent = latent.permute(0, 2, 1, 3, 4).to(dtype=dtype)
            if is_clip_memory:
                latent = latent[:, -1:, :, :, :].contiguous()
            per_frame_latents.append(latent)
        latents.append(torch.cat(per_frame_latents, dim=1))
        del per_frame_latents
    return torch.cat(latents, dim=0)


@torch.no_grad()
def decode_benchmark_sample(video_vae, audio_vae, video_latent, audio_latent):
    video_pixel = video_vae.decode_to_pixel(video_latent)
    audio_waveform = audio_vae.decode_to_waveform(audio_latent) if audio_latent is not None else None

    video_uint8 = video_pixel[0]
    if video_uint8.shape[0] == 3:
        video_uint8 = video_uint8.permute(1, 0, 2, 3)
    video_uint8 = video_uint8.permute(0, 2, 3, 1)
    video_uint8 = (video_uint8.clamp(0, 1) * 255).cpu().to(torch.uint8).contiguous()

    audio_float = normalize_audio_waveform_for_media(audio_waveform)
    return video_uint8, audio_float


def write_benchmark_media(
    *,
    output_path: Path,
    video_uint8: torch.Tensor,
    audio_waveform: Optional[torch.Tensor],
    fps: int,
    audio_sr: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_waveform = normalize_audio_waveform_for_media(audio_waveform)
    stats = audio_waveform_stats(audio_waveform)

    wrote_with_audio = False
    wrote_sidecar_wav = False
    with tempfile.TemporaryDirectory(prefix="echo_media_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        video_only_path = tmp_dir / "video_only.mp4"

        _write_video_only_ffmpeg(video_only_path, video_uint8, fps)

        if audio_waveform is not None:
            audio_tmp_path = tmp_dir / "audio.wav"
            try:
                torchaudio.save(str(audio_tmp_path), audio_waveform, audio_sr)
                _mux_audio_ffmpeg(video_only_path, audio_tmp_path, output_path)
                wrote_with_audio = True
            except Exception as exc:
                print(f"[warn] ffmpeg mux with audio failed for {output_path}: {exc}; audio_stats={stats}", flush=True)

        if not wrote_with_audio:
            shutil.copyfile(video_only_path, output_path)
            if audio_waveform is not None:
                try:
                    torchaudio.save(str(output_path.with_suffix(".wav")), audio_waveform, audio_sr)
                    wrote_sidecar_wav = True
                except Exception as exc:
                    print(f"[warn] torchaudio.save failed for {output_path}: {exc}; audio_stats={stats}", flush=True)

    return {
        "wrote_audio_in_mp4": wrote_with_audio,
        "wrote_sidecar_wav": wrote_sidecar_wav,
        "audio_stats": stats,
    }


def _write_video_only_ffmpeg(output_path: Path, video_uint8: torch.Tensor, fps: int) -> None:
    if video_uint8.ndim != 4 or video_uint8.shape[-1] != 3:
        raise ValueError(f"Expected video tensor shape [T, H, W, 3], got {tuple(video_uint8.shape)}")

    num_frames, height, width, _ = video_uint8.shape
    if num_frames <= 0:
        raise ValueError("Video tensor has no frames to encode")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]

    video_bytes = video_uint8.contiguous().cpu().numpy().tobytes()
    proc = subprocess.run(ffmpeg_cmd, input=video_bytes, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed to encode video-only output:\n{stderr}")


def _mux_audio_ffmpeg(video_path: Path, audio_path: Path, output_path: Path) -> None:
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]
    proc = subprocess.run(ffmpeg_cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed to mux audio:\n{stderr}")


def save_memory_bank_frames(memory_frames: list[Any], save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    for old_file in save_dir.glob("*.jpg"):
        old_file.unlink()
    for idx, frame in enumerate(memory_frames):
        if isinstance(frame, list):
            frame = frame[len(frame) // 2]
        frame.convert("RGB").save(save_dir / f"memory_{idx:03d}.jpg")


def concat_shot_videos(shot_paths: list[Path], output_path: Path) -> None:
    if not shot_paths:
        raise ValueError("No shot videos provided for concatenation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fp:
        concat_file = Path(fp.name)
        for shot_path in shot_paths:
            fp.write(f"file '{shot_path.resolve().as_posix()}'\n")

    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_file), "-c", "copy", str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            fallback_cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                str(output_path),
            ]
            fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
            if fallback_result.returncode != 0:
                raise RuntimeError(
                    "Failed to concatenate shot videos with ffmpeg.\n"
                    f"copy stderr:\n{result.stderr}\n"
                    f"reencode stderr:\n{fallback_result.stderr}"
                )
    finally:
        concat_file.unlink(missing_ok=True)


def concat_shot_audios(audios: list[torch.Tensor]) -> Optional[torch.Tensor]:
    if not audios:
        return None
    audio = audios[0]
    if audio.ndim == 1:
        sample_dim = 0
    elif audio.ndim == 2:
        sample_dim = 1 if audio.shape[0] <= audio.shape[1] else 0
    else:
        raise ValueError(f"Expected audio tensor with 1 or 2 dims, got shape={tuple(audio.shape)}")
    return torch.cat([a.contiguous() for a in audios], dim=sample_dim).contiguous()
