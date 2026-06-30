from __future__ import annotations

import math
import os
import re
import threading
from pathlib import Path
from typing import Any, Protocol

from .audio_service import resolve_static_url


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MIN_TRACK_QUALITY = 0.80
DEFAULT_CLUSTER_SIMILARITY = 0.55
DEFAULT_MAX_TRACKS = 64
CAMPP_MODELSCOPE_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
ECAPA_HF_ID = "speechbrain/spkrec-ecapa-voxceleb"

_BACKEND_LOCK = threading.Lock()
_BACKEND_CACHE: dict[tuple[str, str], SpeakerEmbeddingBackend] = {}
_BACKEND_ERROR_CACHE: dict[tuple[str, str], str] = {}


class SpeakerEmbeddingBackend(Protocol):
    name: str
    status: str
    strong: bool

    def embed(self, track: dict) -> Any | None:
        ...


class LocalSpectralEmbeddingBackend:
    name = "local_spectral"
    status = "degraded_local_embedding"
    strong = False

    def embed(self, track: dict) -> Any | None:
        return _local_spectral_embedding(track)


class EcapaEmbeddingBackend:
    name = "ecapa"
    status = "strong"
    strong = True

    def __init__(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        source = os.getenv("SPEAKER_ECAPA_SOURCE", ECAPA_HF_ID).strip() or ECAPA_HF_ID
        savedir = Path(
            os.getenv(
                "SPEAKER_ECAPA_SAVEDIR",
                str(BACKEND_DIR / "models" / "speechbrain" / "spkrec-ecapa-voxceleb"),
            )
        )
        savedir.mkdir(parents=True, exist_ok=True)
        self._classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": _embedding_device()},
            local_strategy=LocalStrategy.COPY,
        )

    def embed(self, track: dict) -> Any | None:
        try:
            import numpy
            import torch
            import torchaudio.functional as ta_functional

            sample_rate = int(track["sample_rate"])
            samples = _active_embedding_samples(numpy.asarray(track["samples"], dtype="float32"), sample_rate)
            if len(samples) == 0:
                return None
            waveform = torch.from_numpy(samples).float().unsqueeze(0)
            if sample_rate != 16000:
                waveform = ta_functional.resample(waveform, sample_rate, 16000)
            with torch.no_grad():
                embedding = self._classifier.encode_batch(waveform).squeeze().detach().cpu().numpy()
            return _normalize_embedding(embedding)
        except Exception:
            return None


class CamppEmbeddingBackend:
    name = "campp"
    status = "strong"
    strong = True

    def __init__(self) -> None:
        try:
            from funasr import AutoModel
        except Exception as exc:
            raise RuntimeError(f"FunASR CAM++ loader is unavailable: {_short_error(exc)}") from exc

        model_source = os.getenv("SPEAKER_CAMPP_MODEL", CAMPP_MODELSCOPE_ID).strip() or CAMPP_MODELSCOPE_ID
        local_model = BACKEND_DIR / ".runtime" / "modelscope" / "models" / "iic" / "speech_campplus_sv_zh-cn_16k-common"
        if local_model.exists() and not os.getenv("SPEAKER_CAMPP_MODEL"):
            model_source = str(local_model)
        self._model = AutoModel(model=model_source, device=_embedding_device(), disable_update=True)

    def embed(self, track: dict) -> Any | None:
        try:
            result = self._model.generate(input=str(track["path"]))
            return _normalize_embedding(_extract_embedding_from_funasr(result))
        except Exception:
            return None


def add_speaker_count_estimation(separation: dict) -> dict:
    """Attach post-separation speaker-count diagnostics without steering separation."""

    estimation = estimate_speaker_count_from_tracks(separation.get("tracks", []) or [])
    return {
        **separation,
        "speaker_count_estimation": estimation,
        "metrics": {
            **separation.get("metrics", {}),
            **speaker_count_estimation_metrics(estimation),
        },
    }


def estimate_speaker_count_from_tracks(separated_tracks: list[dict], *, max_tracks: int | None = None) -> dict:
    if not _estimation_enabled():
        return {
            "status": "disabled",
            "estimated_speaker_count": 0,
            "global_estimated_speaker_count": 0,
            "tracks": [],
            "clusters": [],
        }

    loaded = _load_tracks(separated_tracks[: (max_tracks or _max_tracks())])
    if not loaded:
        return _empty_result("no_valid_tracks", len(separated_tracks))

    backend, backend_error = _select_embedding_backend()
    if backend is None:
        return {
            **_empty_result("embedding_backend_unavailable", len(separated_tracks)),
            "embedding_backend": _requested_backend(),
            "embedding_backend_status": "unavailable",
            "embedding_backend_error": backend_error,
        }

    correlations = _pairwise_correlations(loaded)
    scored_tracks = []
    for index, track in enumerate(loaded):
        quality = _score_track_quality(track, correlations[index])
        embedding = _normalize_embedding(backend.embed(track))
        embedding_quality = 1.0 if embedding is not None else 0.0
        scored_tracks.append(
            {
                "track_id": track["track_id"],
                "label": track["label"],
                "path": str(track["path"]),
                "duration_seconds": round(float(track["duration_seconds"]), 3),
                "active_ratio": round(float(track["active_ratio"]), 4),
                "rms": round(float(track["rms"]), 6),
                "max_abs_correlation": round(float(max(correlations[index]) if correlations[index] else 0.0), 4),
                "quality_score": round(float(quality), 4),
                "embedding_quality": round(float(embedding_quality), 4),
                "accepted": bool(quality >= _min_track_quality() and embedding is not None),
                "embedding": embedding,
            }
        )

    accepted = [track for track in scored_tracks if track["accepted"]]
    if not accepted:
        return {
            **_empty_result("no_accepted_tracks", len(separated_tracks)),
            "embedding_backend": backend.name,
            "embedding_backend_status": backend.status,
            "candidate_track_count": len(scored_tracks),
            "tracks": [_public_track(track) for track in scored_tracks],
        }

    clusters = _cluster_tracks(accepted)
    raw_cluster_count = len(clusters)
    single_window_tracks = _should_keep_single_window_tracks(accepted, clusters)
    if single_window_tracks:
        clusters = [[track] for track in accepted]
    _assign_global_speaker_ids(clusters)
    stability = _cluster_stability(accepted, clusters)
    public_tracks = [_public_track(track) for track in scored_tracks]
    public_clusters = [_public_cluster(index, cluster) for index, cluster in enumerate(clusters, start=1) if cluster]
    min_quality = min((float(track["quality_score"]) for track in accepted), default=0.0)
    status = "ok" if backend.strong else "degraded_local_embedding"
    raw_count = raw_cluster_count if backend.strong else 0
    window_counts = _window_cluster_counts(accepted) if backend.strong else {}
    count = len(accepted) if single_window_tracks and backend.strong else _resolve_global_count(raw_count, window_counts) if backend.strong else 0
    count_source = "single_window_tracks" if single_window_tracks else "window_consensus" if window_counts and count != raw_count else "global_clusters"
    return {
        "status": status,
        "algorithm": "quality_filter+speaker_embedding+agglomerative_cross_window_clustering",
        "embedding_backend": backend.name,
        "embedding_backend_status": backend.status,
        "estimated_speaker_count": count,
        "global_estimated_speaker_count": count,
        "raw_global_estimated_speaker_count": raw_count,
        "window_estimated_speaker_counts": window_counts,
        "global_count_source": count_source,
        "candidate_track_count": len(scored_tracks),
        "accepted_track_count": len(accepted),
        "min_track_quality": round(min_quality, 4),
        "cluster_stability": round(float(stability), 4),
        "stability_score": round(float(stability), 4),
        "cluster_similarity_threshold": _cluster_similarity_threshold(),
        "tracks": public_tracks,
        "clusters": public_clusters,
    }


def speaker_count_estimation_metrics(estimation: dict) -> dict[str, str]:
    status = str(estimation.get("status", "missing"))
    metrics = {
        "speaker_count_estimation_status": status,
        "estimated_speaker_count": str(estimation.get("estimated_speaker_count", 0)),
        "speaker_count_embedding_backend": str(estimation.get("embedding_backend", "")),
        "speaker_count_embedding_backend_status": str(estimation.get("embedding_backend_status", "")),
    }
    if estimation.get("embedding_backend_error"):
        metrics["speaker_count_embedding_backend_error"] = str(estimation.get("embedding_backend_error"))
    if status not in {"ok", "degraded_local_embedding"}:
        return metrics

    metrics.update(
        {
            "speaker_count_estimation_algorithm": str(estimation.get("algorithm", "")),
            "speaker_count_cluster_stability": f"{float(estimation.get('cluster_stability', 0.0)):.3f}",
            "speaker_count_estimation_stability": f"{float(estimation.get('stability_score', 0.0)):.3f}",
            "speaker_count_min_track_quality": f"{float(estimation.get('min_track_quality', 0.0)):.3f}",
            "speaker_count_estimation_min_track_quality": f"{float(estimation.get('min_track_quality', 0.0)):.3f}",
            "speaker_count_estimation_accepted_tracks": str(estimation.get("accepted_track_count", 0)),
            "speaker_count_estimation_candidate_tracks": str(estimation.get("candidate_track_count", 0)),
            "speaker_count_global_cluster_summary": _cluster_summary(estimation.get("clusters", []) or []),
            "speaker_count_cluster_summary": _cluster_summary(estimation.get("clusters", []) or []),
            "speaker_count_global_count_source": str(estimation.get("global_count_source", "")),
            "speaker_count_raw_global_estimated_speaker_count": str(estimation.get("raw_global_estimated_speaker_count", "")),
            "speaker_count_window_estimated_speaker_counts": _window_count_summary(estimation.get("window_estimated_speaker_counts", {}) or {}),
        }
    )
    for index, track in enumerate(estimation.get("tracks", []) or [], start=1):
        metrics[f"speaker_count_track_{index}_quality"] = (
            f"{track.get('label', track.get('track_id', index))}:"
            f"{float(track.get('quality_score', 0.0)):.3f}/"
            f"{'accepted' if track.get('accepted') else 'filtered'}"
        )
        if track.get("global_speaker_id"):
            metrics[f"speaker_count_track_{index}_global_speaker_id"] = str(track.get("global_speaker_id"))
    return metrics


def _select_embedding_backend() -> tuple[SpeakerEmbeddingBackend | None, str]:
    requested = _requested_backend()
    candidates = _backend_candidates(requested)
    errors = []
    for name in candidates:
        cache_key = (name, _embedding_device())
        with _BACKEND_LOCK:
            if cache_key in _BACKEND_CACHE:
                return _BACKEND_CACHE[cache_key], ""
            if cache_key in _BACKEND_ERROR_CACHE:
                errors.append(f"{name}: {_BACKEND_ERROR_CACHE[cache_key]}")
                continue
        try:
            backend = _create_backend(name)
        except Exception as exc:
            message = _short_error(exc)
            with _BACKEND_LOCK:
                _BACKEND_ERROR_CACHE[cache_key] = message
            errors.append(f"{name}: {message}")
            continue
        if _strong_required() and not backend.strong:
            errors.append(f"{name}: local spectral backend is degraded and strong embedding is required")
            continue
        with _BACKEND_LOCK:
            _BACKEND_CACHE[cache_key] = backend
        return backend, ""
    return None, "; ".join(errors) if errors else "no backend candidates"


def _create_backend(name: str) -> SpeakerEmbeddingBackend:
    if name == "campp":
        return CamppEmbeddingBackend()
    if name == "ecapa":
        return EcapaEmbeddingBackend()
    if name == "local_spectral":
        return LocalSpectralEmbeddingBackend()
    raise RuntimeError(f"Unsupported speaker embedding backend: {name}")


def _backend_candidates(requested: str) -> list[str]:
    if requested == "auto":
        return ["campp", "ecapa"] if _strong_required() else ["campp", "ecapa", "local_spectral"]
    return [requested]


def _load_tracks(separated_tracks: list[dict]) -> list[dict]:
    loaded = []
    for index, track in enumerate(separated_tracks, start=1):
        path = resolve_static_url(str(track.get("audio_url", "")))
        if path is None or not path.exists():
            continue
        try:
            import numpy
            import soundfile

            data, sample_rate = soundfile.read(str(path), always_2d=True, dtype="float32")
            if len(data) == 0:
                continue
            samples = data.mean(axis=1).astype("float32")
            rms_values = _frame_rms(samples, int(sample_rate), numpy)
            active_threshold = max(1e-5, float(numpy.percentile(rms_values, 75)) * 0.25) if len(rms_values) else 1e-5
            loaded.append(
                {
                    "track_id": str(track.get("track_id") or path.stem or f"track_{index}"),
                    "label": str(track.get("label") or track.get("track_id") or f"track {index}"),
                    "path": path,
                    "samples": samples,
                    "sample_rate": int(sample_rate),
                    "duration_seconds": float(len(samples) / max(1, int(sample_rate))),
                    "rms": float(math.sqrt(float(numpy.mean(samples * samples)) + 1e-12)),
                    "active_ratio": float(numpy.mean(rms_values >= active_threshold)) if len(rms_values) else 0.0,
                    "clipping_ratio": float(numpy.mean(numpy.abs(samples) >= 0.98)) if len(samples) else 0.0,
                }
            )
        except Exception:
            continue
    return loaded


def _score_track_quality(track: dict, correlations: list[float]) -> float:
    max_corr = max(correlations) if correlations else 0.0
    activity_score = min(1.0, float(track["active_ratio"]) / 0.25)
    rms_score = min(1.0, float(track["rms"]) / 0.015)
    isolation_score = max(0.0, 1.0 - min(0.75, max_corr) / 0.75)
    duration_score = min(1.0, float(track["duration_seconds"]) / 1.0)
    clipping_score = max(0.0, 1.0 - min(0.02, float(track["clipping_ratio"])) / 0.02)
    return max(
        0.0,
        min(
            1.0,
            0.30 * activity_score
            + 0.25 * rms_score
            + 0.30 * isolation_score
            + 0.10 * duration_score
            + 0.05 * clipping_score,
        ),
    )


def _local_spectral_embedding(track: dict) -> Any | None:
    try:
        import numpy

        samples = track["samples"]
        sample_rate = int(track["sample_rate"])
        frame = max(512, int(sample_rate * 0.064))
        hop = max(256, frame // 2)
        if len(samples) < frame:
            samples = numpy.pad(samples, (0, frame - len(samples)))

        rms_values = []
        frames = []
        for start in range(0, max(1, len(samples) - frame + 1), hop):
            segment = samples[start : start + frame]
            if len(segment) < frame:
                segment = numpy.pad(segment, (0, frame - len(segment)))
            rms = float(numpy.sqrt(numpy.mean(segment * segment) + 1e-12))
            rms_values.append(rms)
            frames.append(segment)
        if not frames:
            return None

        active_threshold = max(1e-5, float(numpy.percentile(rms_values, 75)) * 0.25)
        active_frames = [frame_data for frame_data, rms in zip(frames, rms_values) if rms >= active_threshold]
        if not active_frames:
            return None

        window = numpy.hanning(frame).astype("float32")
        freqs = numpy.fft.rfftfreq(frame, d=1.0 / float(sample_rate))
        band_features = []
        zcr_values = []
        centroid_values = []
        bandwidth_values = []
        for frame_data in active_frames:
            spectrum = numpy.log1p(numpy.abs(numpy.fft.rfft(frame_data * window))).astype("float32")
            band_edges = numpy.linspace(0, len(spectrum), 33, dtype=int)
            band_features.append(
                [
                    float(numpy.mean(spectrum[start : max(end, start + 1)]))
                    for start, end in zip(band_edges[:-1], band_edges[1:])
                ]
            )
            zcr_values.append(float(numpy.mean(numpy.abs(numpy.diff(numpy.signbit(frame_data))))))
            weight_sum = float(numpy.sum(spectrum) + 1e-9)
            centroid = float(numpy.sum(freqs * spectrum) / weight_sum)
            centroid_values.append(centroid / max(1.0, sample_rate / 2.0))
            bandwidth_values.append(float(numpy.sqrt(numpy.sum(((freqs - centroid) ** 2) * spectrum) / weight_sum)) / max(1.0, sample_rate / 2.0))

        features = numpy.asarray(
            list(numpy.mean(numpy.asarray(band_features), axis=0))
            + [
                float(numpy.mean(rms_values)),
                float(numpy.std(rms_values)),
                float(numpy.mean(zcr_values)),
                float(numpy.std(zcr_values)),
                float(numpy.mean(centroid_values)),
                float(numpy.std(centroid_values)),
                float(numpy.mean(bandwidth_values)),
                float(numpy.std(bandwidth_values)),
            ],
            dtype="float32",
        )
        features = numpy.log1p(numpy.maximum(features, 0.0))
        return _normalize_embedding(features)
    except Exception:
        return None


def _active_embedding_samples(samples: Any, sample_rate: int) -> Any:
    try:
        import numpy

        samples = numpy.asarray(samples, dtype="float32")
        if len(samples) == 0:
            return samples
        frame = max(1, int(sample_rate * 0.50))
        hop = max(1, int(sample_rate * 0.25))
        if len(samples) <= frame:
            return samples
        windows = []
        for start in range(0, len(samples) - frame + 1, hop):
            segment = samples[start : start + frame]
            rms = float(numpy.sqrt(numpy.mean(segment * segment) + 1e-12))
            windows.append((start, rms))
        if not windows:
            return samples
        rms_values = numpy.asarray([item[1] for item in windows], dtype="float32")
        threshold = max(1e-5, float(numpy.percentile(rms_values, 70)) * 0.45)
        selected = [(start, rms) for start, rms in windows if rms >= threshold]
        if not selected:
            selected = sorted(windows, key=lambda item: item[1], reverse=True)[:4]
        max_windows = max(4, int(float(os.getenv("SPEAKER_EMBEDDING_MAX_ACTIVE_SECONDS", "12")) / 0.5))
        selected = sorted(selected, key=lambda item: item[1], reverse=True)[:max_windows]
        selected = sorted(selected, key=lambda item: item[0])
        output = numpy.concatenate([samples[start : start + frame] for start, _rms in selected]).astype("float32")
        return output if len(output) else samples
    except Exception:
        return samples


def _cluster_tracks(tracks: list[dict]) -> list[list[dict]]:
    clusters = [[track] for track in tracks]
    threshold = _cluster_similarity_threshold()
    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_score = -1.0
        for first_index in range(len(clusters)):
            for second_index in range(first_index + 1, len(clusters)):
                score = _cluster_similarity(clusters[first_index], clusters[second_index])
                if score > best_score:
                    best_score = score
                    best_pair = (first_index, second_index)
        if best_pair is None or best_score < threshold:
            break
        first_index, second_index = best_pair
        clusters[first_index].extend(clusters[second_index])
        del clusters[second_index]
    return clusters


def _window_cluster_counts(tracks: list[dict]) -> dict[str, int]:
    grouped: dict[str, list[dict]] = {}
    for track in tracks:
        window_key = _window_key(str(track.get("track_id", "")))
        if window_key:
            grouped.setdefault(window_key, []).append(track)
    if len(grouped) <= 1:
        return {}
    return {key: len(_cluster_tracks(value)) for key, value in sorted(grouped.items()) if value}


def _should_keep_single_window_tracks(tracks: list[dict], clusters: list[list[dict]]) -> bool:
    if any(_window_key(str(track.get("track_id", ""))) for track in tracks):
        return False
    if len(tracks) <= 1 or len(clusters) >= len(tracks):
        return False
    if len(tracks) > _single_window_track_count_max():
        return False
    min_quality = min(float(track.get("quality_score", 0.0)) for track in tracks)
    return min_quality >= _min_track_quality()


def _resolve_global_count(raw_count: int, window_counts: dict[str, int]) -> int:
    if not window_counts:
        return raw_count
    max_window_count = max(window_counts.values(), default=0)
    if max_window_count <= 0:
        return raw_count
    if raw_count <= 0:
        return max_window_count
    return max_window_count if raw_count != max_window_count else raw_count


def _window_key(track_id: str) -> str:
    match = re.match(r"^(w\d+)_", track_id)
    return match.group(1) if match else ""


def _cluster_similarity(first: list[dict], second: list[dict]) -> float:
    scores = []
    for left in first:
        for right in second:
            scores.append(_embedding_similarity(left.get("embedding"), right.get("embedding")))
    return max(scores) if scores else 0.0


def _pairwise_correlations(tracks: list[dict]) -> list[list[float]]:
    output: list[list[float]] = []
    for first_index, first in enumerate(tracks):
        row = []
        for second_index, second in enumerate(tracks):
            if first_index == second_index:
                continue
            row.append(_waveform_correlation(first, second))
        output.append(row)
    return output


def _waveform_correlation(first: dict, second: dict) -> float:
    try:
        import numpy

        left = first["samples"]
        right = second["samples"]
        size = min(len(left), len(right))
        if size <= 16:
            return 0.0
        left = left[:size] - float(numpy.mean(left[:size]))
        right = right[:size] - float(numpy.mean(right[:size]))
        denom = float(numpy.linalg.norm(left) * numpy.linalg.norm(right))
        if denom <= 1e-9:
            return 0.0
        return abs(float(numpy.dot(left, right) / denom))
    except Exception:
        return 0.0


def _embedding_similarity(first: Any | None, second: Any | None) -> float:
    if first is None or second is None:
        return 0.0
    try:
        import numpy

        return float(numpy.clip(numpy.dot(first, second), -1.0, 1.0))
    except Exception:
        return 0.0


def _cluster_stability(tracks: list[dict], clusters: list[list[dict]]) -> float:
    if not tracks:
        return 0.0
    min_quality = min(float(track["quality_score"]) for track in tracks)
    if len(clusters) <= 1:
        separation_margin = 1.0
    else:
        max_inter_similarity = 0.0
        for first_index in range(len(clusters)):
            for second_index in range(first_index + 1, len(clusters)):
                max_inter_similarity = max(max_inter_similarity, _cluster_similarity(clusters[first_index], clusters[second_index]))
        separation_margin = max(0.0, 1.0 - max_inter_similarity)
    embedding_coverage = sum(1 for track in tracks if track.get("embedding") is not None) / len(tracks)
    cluster_scores = [float(cluster.get("stability_score", 0.0)) for cluster in (_public_cluster(i, c) for i, c in enumerate(clusters, start=1))]
    cluster_score = sum(cluster_scores) / len(cluster_scores) if cluster_scores else 0.0
    return max(0.0, min(1.0, 0.35 * min_quality + 0.25 * separation_margin + 0.25 * embedding_coverage + 0.15 * cluster_score))


def _public_cluster(index: int, cluster: list[dict]) -> dict:
    similarities = [
        _embedding_similarity(left.get("embedding"), right.get("embedding"))
        for left_index, left in enumerate(cluster)
        for right in cluster[left_index + 1 :]
    ]
    if similarities:
        mean_similarity = sum(similarities) / len(similarities)
    else:
        mean_similarity = 1.0
    mean_quality = sum(float(track["quality_score"]) for track in cluster) / len(cluster)
    stability = max(0.0, min(1.0, 0.55 * mean_quality + 0.45 * max(0.0, mean_similarity)))
    global_speaker_id = f"speaker_{index:02d}"
    return {
        "cluster_id": global_speaker_id,
        "global_speaker_id": global_speaker_id,
        "track_ids": [track["track_id"] for track in cluster],
        "mean_quality": round(float(mean_quality), 4),
        "mean_similarity": round(float(mean_similarity), 4),
        "stability_score": round(float(stability), 4),
    }


def _assign_global_speaker_ids(clusters: list[list[dict]]) -> None:
    clusters.sort(key=lambda cluster: min(str(track["track_id"]) for track in cluster))
    for index, cluster in enumerate(clusters, start=1):
        speaker_id = f"speaker_{index:02d}"
        for track in cluster:
            track["global_speaker_id"] = speaker_id
            track["track_cluster_id"] = speaker_id


def _frame_rms(samples: Any, sample_rate: int, numpy: Any) -> Any:
    frame = max(256, int(sample_rate * 0.064))
    hop = max(128, frame // 2)
    if len(samples) < frame:
        return numpy.asarray([float(numpy.sqrt(numpy.mean(samples * samples) + 1e-12))], dtype="float32")
    return numpy.asarray(
        [
            float(numpy.sqrt(numpy.mean(samples[index : index + frame] ** 2) + 1e-12))
            for index in range(0, len(samples) - frame + 1, hop)
        ],
        dtype="float32",
    )


def _public_track(track: dict) -> dict:
    return {
        "track_id": track["track_id"],
        "label": track["label"],
        "duration_seconds": track["duration_seconds"],
        "active_ratio": track["active_ratio"],
        "rms": track["rms"],
        "max_abs_correlation": track["max_abs_correlation"],
        "quality_score": track["quality_score"],
        "embedding_quality": track.get("embedding_quality", 0.0),
        "accepted": track["accepted"],
        "track_cluster_id": track.get("track_cluster_id"),
        "global_speaker_id": track.get("global_speaker_id"),
    }


def _cluster_summary(clusters: list[dict]) -> str:
    if not clusters:
        return "none"
    return "; ".join(
        f"{cluster.get('global_speaker_id', cluster.get('cluster_id', index))}={len(cluster.get('track_ids', []) or [])} track(s),"
        f" sim={float(cluster.get('mean_similarity', 0.0)):.3f}, stable={float(cluster.get('stability_score', 0.0)):.3f}"
        for index, cluster in enumerate(clusters, start=1)
    )


def _window_count_summary(window_counts: dict[str, int]) -> str:
    if not window_counts:
        return "none"
    return "; ".join(f"{key}={value}" for key, value in sorted(window_counts.items()))


def _empty_result(status: str, candidate_count: int) -> dict:
    return {
        "status": status,
        "estimated_speaker_count": 0,
        "global_estimated_speaker_count": 0,
        "candidate_track_count": candidate_count,
        "accepted_track_count": 0,
        "embedding_backend": "",
        "embedding_backend_status": "",
        "tracks": [],
        "clusters": [],
    }


def _extract_embedding_from_funasr(result: Any) -> Any | None:
    if isinstance(result, dict):
        for key in ("embedding", "spk_embedding", "vector", "xvector"):
            if key in result:
                return result[key]
    if isinstance(result, list):
        for item in result:
            value = _extract_embedding_from_funasr(item)
            if value is not None:
                return value
    return None


def _normalize_embedding(value: Any | None) -> Any | None:
    if value is None:
        return None
    try:
        import numpy

        embedding = numpy.asarray(value, dtype="float32").reshape(-1)
        if embedding.size == 0 or not bool(numpy.all(numpy.isfinite(embedding))):
            return None
        norm = float(numpy.linalg.norm(embedding))
        if norm <= 1e-9:
            return None
        return embedding / norm
    except Exception:
        return None


def _short_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:220] or exc.__class__.__name__


def _estimation_enabled() -> bool:
    return os.getenv("SEPARATION_SPEAKER_COUNT_ESTIMATION", "true").strip().lower() not in {"0", "false", "no", "off"}


def _requested_backend() -> str:
    return os.getenv("SPEAKER_EMBEDDING_BACKEND", "ecapa").strip().lower() or "ecapa"


def _strong_required() -> bool:
    return os.getenv("SPEAKER_EMBEDDING_STRONG_REQUIRED", "true").strip().lower() not in {"0", "false", "no", "off"}


def _embedding_device() -> str:
    return os.getenv("SPEAKER_EMBEDDING_DEVICE", "cpu").strip().lower() or "cpu"


def _min_track_quality() -> float:
    return _float_env("SPEAKER_COUNT_MIN_TRACK_QUALITY", DEFAULT_MIN_TRACK_QUALITY, minimum=0.0, maximum=1.0)


def _cluster_similarity_threshold() -> float:
    if os.getenv("SPEAKER_CLUSTER_THRESHOLD"):
        return _float_env("SPEAKER_CLUSTER_THRESHOLD", DEFAULT_CLUSTER_SIMILARITY, minimum=-1.0, maximum=1.0)
    return _float_env("SPEAKER_COUNT_CLUSTER_SIMILARITY", DEFAULT_CLUSTER_SIMILARITY, minimum=-1.0, maximum=1.0)


def _max_tracks() -> int:
    raw = os.getenv("SPEAKER_COUNT_MAX_TRACKS", str(DEFAULT_MAX_TRACKS)).strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_TRACKS
    return max(1, min(256, value))


def _single_window_track_count_max() -> int:
    raw = os.getenv("SPEAKER_COUNT_SINGLE_WINDOW_TRACK_COUNT_MAX", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(1, min(32, value))


def _float_env(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError:
        return default
    return min(maximum, max(minimum, value))
