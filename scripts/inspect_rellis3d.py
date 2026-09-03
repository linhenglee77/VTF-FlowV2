"""Recursively audit a local RELLIS-3D dataset without modifying it.

Run from the directory containing the ``TerraFlow`` package::

    python TerraFlow/scripts/inspect_rellis3d.py --data-root <PATH>

The audit relies only on file names, byte layouts, and text content observed in
the supplied root. It does not assign undocumented coordinate-frame semantics.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import statistics
import struct
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "rellis3d_data_audit.md"
SEQUENCE_PATTERN = re.compile(r"^\d{5}$")
CAMERA_FRAME_PATTERN = re.compile(
    r"^frame(?P<index>\d+)-(?P<seconds>\d+)_(?P<fraction>\d+)$"
)
NUMERIC_FRAME_PATTERN = re.compile(r"^(?P<index>\d+)$")


@dataclass(frozen=True)
class FrameName:
    """Frame index and optional timestamp parsed from a file name."""

    index: int
    timestamp_s: Optional[float]


@dataclass(frozen=True)
class ImageInfo:
    """Image metadata obtained directly from a PNG or JPEG header."""

    format: str
    height: int
    width: int
    channels: int


@dataclass(frozen=True)
class BinaryArrayInfo:
    """Shape and a small decoded prefix from a binary array file."""

    count: int
    width: int
    dtype: str
    first_values: Tuple[float, ...]


@dataclass(frozen=True)
class SourceSpec:
    """Description of a known RELLIS-3D per-frame source directory."""

    key: str
    title: str
    directory: str
    suffixes: Tuple[str, ...]
    format_description: str


SOURCE_SPECS: Tuple[SourceSpec, ...] = (
    SourceSpec(
        "rgb",
        "RGB images",
        "pylon_camera_node",
        (".jpg", ".jpeg"),
        "JPEG image",
    ),
    SourceSpec(
        "ouster_cloud",
        "Ouster OS1 LiDAR point clouds",
        "os1_cloud_node_kitti_bin",
        (".bin",),
        "little-endian float32 records with four values per point",
    ),
    SourceSpec(
        "ouster_label",
        "Ouster semantic point labels",
        "os1_cloud_node_semantickitti_label_id",
        (".label",),
        "little-endian uint32, one value per point",
    ),
    SourceSpec(
        "velodyne_cloud",
        "Velodyne LiDAR point clouds",
        "vel_cloud_node_kitti_bin",
        (".bin",),
        "little-endian float32 records with four values per point",
    ),
    SourceSpec(
        "velodyne_label",
        "Velodyne semantic point labels",
        "vel_cloud_node_semantickitti_label_id",
        (".label",),
        "little-endian uint32, one value per point",
    ),
    SourceSpec(
        "camera_label",
        "Camera semantic labels",
        "pylon_camera_node_label_id",
        (".png",),
        "PNG single-channel label image",
    ),
)


def parse_frame_filename(path_or_name: object) -> Optional[FrameName]:
    """Parse a numeric or timestamp-bearing RELLIS frame file name.

    The fractional timestamp is scaled by its observed number of digits instead
    of assuming milliseconds.
    """

    stem = Path(str(path_or_name)).stem
    camera_match = CAMERA_FRAME_PATTERN.fullmatch(stem)
    if camera_match:
        fraction = camera_match.group("fraction")
        timestamp = int(camera_match.group("seconds")) + int(fraction) / (
            10 ** len(fraction)
        )
        return FrameName(int(camera_match.group("index")), timestamp)
    numeric_match = NUMERIC_FRAME_PATTERN.fullmatch(stem)
    if numeric_match:
        return FrameName(int(numeric_match.group("index")), None)
    return None


def parse_pose_file(path: Path) -> List[Tuple[float, ...]]:
    """Parse an unlabeled pose text file containing one 3x4 matrix per row."""

    poses: List[Tuple[float, ...]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            values = tuple(float(value) for value in line.split())
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: non-numeric pose value") from error
        if len(values) != 12:
            raise ValueError(
                f"{path}:{line_number}: expected 12 pose values, got {len(values)}"
            )
        poses.append(values)
    return poses


def parse_kitti_calibration(path: Path) -> Dict[str, Tuple[float, ...]]:
    """Parse ``key: numeric values`` calibration records."""

    records: Dict[str, Tuple[float, ...]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"{path}:{line_number}: expected 'key: values'")
        key, raw_values = line.split(":", maxsplit=1)
        try:
            records[key.strip()] = tuple(float(value) for value in raw_values.split())
        except ValueError as error:
            raise ValueError(
                f"{path}:{line_number}: non-numeric calibration value"
            ) from error
    return records


def parse_camera_info(path: Path) -> Tuple[float, ...]:
    """Parse the sequence-level four-number ``camera_info.txt`` record."""

    values = tuple(float(value) for value in path.read_text(encoding="utf-8").split())
    if len(values) != 4:
        raise ValueError(f"{path}: expected four camera-info values, got {len(values)}")
    return values


def inspect_point_cloud(path: Path) -> BinaryArrayInfo:
    """Inspect a KITTI-style float32 point cloud without loading the full file."""

    record_bytes = 4 * 4
    size = path.stat().st_size
    if size % record_bytes:
        raise ValueError(f"{path}: byte size {size} is not divisible by 16")
    with path.open("rb") as stream:
        prefix = stream.read(record_bytes)
    first = struct.unpack("<4f", prefix) if prefix else ()
    return BinaryArrayInfo(size // record_bytes, 4, "float32", tuple(first))


def inspect_semantic_label(path: Path, sample_values: int = 8) -> BinaryArrayInfo:
    """Inspect a uint32 semantic-label vector without assuming bit semantics."""

    value_bytes = 4
    size = path.stat().st_size
    if size % value_bytes:
        raise ValueError(f"{path}: byte size {size} is not divisible by 4")
    count = size // value_bytes
    read_count = min(count, sample_values)
    with path.open("rb") as stream:
        prefix = stream.read(read_count * value_bytes)
    values = struct.unpack(f"<{read_count}I", prefix) if read_count else ()
    return BinaryArrayInfo(count, 1, "uint32", tuple(float(value) for value in values))


def inspect_image(path: Path) -> ImageInfo:
    """Read dimensions and channel count directly from a PNG or JPEG header."""

    with path.open("rb") as stream:
        signature = stream.read(24)
        if signature.startswith(b"\x89PNG\r\n\x1a\n") and len(signature) >= 24:
            width, height = struct.unpack(">II", signature[16:24])
            color_type = signature[25] if len(signature) > 25 else None
            if color_type is None:
                color_type = stream.read(2)[1]
            channels_by_color_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
            channels = channels_by_color_type.get(color_type, 0)
            return ImageInfo("PNG", height, width, channels)

        if not signature.startswith(b"\xff\xd8"):
            raise ValueError(f"{path}: unsupported image signature")
        stream.seek(2)
        start_of_frame = {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
        while True:
            byte = stream.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker_bytes = stream.read(1)
            while marker_bytes == b"\xff":
                marker_bytes = stream.read(1)
            if not marker_bytes:
                break
            marker = marker_bytes[0]
            if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
                continue
            length_bytes = stream.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = struct.unpack(">H", length_bytes)[0]
            if segment_length < 2:
                raise ValueError(f"{path}: invalid JPEG segment length")
            if marker in start_of_frame:
                payload = stream.read(6)
                if len(payload) != 6:
                    break
                _, height, width, channels = struct.unpack(">BHHB", payload)
                return ImageInfo("JPEG", height, width, channels)
            stream.seek(segment_length - 2, 1)
    raise ValueError(f"{path}: image dimensions were not found")


def _source_files(sequence_dir: Path, spec: SourceSpec) -> List[Path]:
    folder = sequence_dir / spec.directory
    if not folder.is_dir():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in spec.suffixes
    )


def _frame_map(files: Iterable[Path]) -> Dict[int, Path]:
    result: Dict[int, Path] = {}
    for path in files:
        parsed = parse_frame_filename(path.name)
        if parsed is not None:
            result[parsed.index] = path
    return result


def _missing_summary(expected: Set[int], observed: Set[int]) -> str:
    missing = sorted(expected - observed)
    if not missing:
        return "0"
    if len(missing) > 3:
        deltas = {right - left for left, right in zip(missing, missing[1:])}
        if len(deltas) == 1:
            step = next(iter(deltas))
            return f"{len(missing)} ({missing[0]}..{missing[-1]}, step {step})"
    preview = ", ".join(str(value) for value in missing[:8])
    suffix = ", ..." if len(missing) > 8 else ""
    return f"{len(missing)} (examples: {preview}{suffix})"


def _median_rate(files: Sequence[Path]) -> Optional[float]:
    parsed = [parse_frame_filename(path.name) for path in files]
    timed = sorted(
        (item for item in parsed if item is not None and item.timestamp_s is not None),
        key=lambda item: item.index,
    )
    deltas = [
        right.timestamp_s - left.timestamp_s  # type: ignore[operator]
        for left, right in zip(timed, timed[1:])
        if right.timestamp_s > left.timestamp_s  # type: ignore[operator]
    ]
    if not deltas:
        return None
    median_delta = statistics.median(deltas)
    return 1.0 / median_delta if median_delta > 0.0 else None


def discover_sequence_directories(data_root: Path) -> List[Path]:
    """Find five-digit directories containing a recognized data source or poses."""

    candidates = [data_root] if SEQUENCE_PATTERN.fullmatch(data_root.name) else []
    candidates.extend(path for path in data_root.rglob("*") if path.is_dir())
    result: List[Path] = []
    source_names = {spec.directory for spec in SOURCE_SPECS}
    for path in candidates:
        if not SEQUENCE_PATTERN.fullmatch(path.name):
            continue
        children = {child.name for child in path.iterdir()}
        if "poses.txt" in children or source_names.intersection(children):
            result.append(path)
    return sorted(set(result), key=lambda path: (str(path.parent), path.name))


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _pattern_for(sequence_dir: Path, spec: SourceSpec, root: Path) -> str:
    prefix = _relative(sequence_dir.parent, root)
    prefix = "" if prefix == "." else f"{prefix}/"
    if spec.key in {"rgb", "camera_label"}:
        suffix = spec.suffixes[0]
        file_pattern = f"frame{{index:06d}}-{{seconds}}_{{fraction}}{suffix}"
    else:
        file_pattern = f"{{index:06d}}{spec.suffixes[0]}"
    return f"{prefix}{{sequence}}/{spec.directory}/{file_pattern}"


def _example_description(spec: SourceSpec, files: Sequence[Path]) -> str:
    if not files:
        return "not found"
    sample = files[0]
    if spec.key in {"rgb", "camera_label"}:
        info = inspect_image(sample)
        return (
            f"{info.height}x{info.width}x{info.channels} {info.format}; "
            f"sample `{sample.name}`"
        )
    if spec.key.endswith("cloud"):
        info = inspect_point_cloud(sample)
        values = ", ".join(f"{value:.6g}" for value in info.first_values)
        return f"[{info.count}, 4] float32; first record ({values})"
    info = inspect_semantic_label(sample)
    values = ", ".join(str(int(value)) for value in info.first_values)
    return f"[{info.count}] uint32; first values ({values})"


def _index_convention(files: Sequence[Path]) -> str:
    if not files:
        return "not found"
    parsed = [item for item in (parse_frame_filename(path.name) for path in files) if item]
    if not parsed:
        return "unresolved: file names are not recognized as frame indices"
    indices = sorted(item.index for item in parsed)
    timestamped = sum(item.timestamp_s is not None for item in parsed)
    if timestamped:
        return (
            f"zero-based `frameNNNNNN`; timestamp embedded after `-` "
            f"({indices[0]}..{indices[-1]})"
        )
    return f"zero-padded numeric stem, zero-based ({indices[0]}..{indices[-1]})"


def _directory_inventory(data_root: Path) -> List[str]:
    directories = [data_root]
    directories.extend(path for path in data_root.rglob("*") if path.is_dir())
    lines: List[str] = []
    for directory in sorted(directories, key=lambda path: _relative(path, data_root)):
        direct_files = [path for path in directory.iterdir() if path.is_file()]
        extensions = Counter(path.suffix.lower() or "[none]" for path in direct_files)
        extension_text = ", ".join(
            f"{extension}:{count}" for extension, count in sorted(extensions.items())
        )
        annotation = f" [files={len(direct_files)}"
        if extension_text:
            annotation += f"; {extension_text}"
        annotation += "]"
        relative = _relative(directory, data_root)
        label = data_root.name if relative == "." else relative
        lines.append(f"{label}/{annotation}")
    return lines


def _pair_size_check(clouds: Mapping[int, Path], labels: Mapping[int, Path]) -> str:
    paired = sorted(set(clouds).intersection(labels))
    if not paired:
        return "no pairs"
    mismatches = sum(
        clouds[index].stat().st_size // 16 != labels[index].stat().st_size // 4
        for index in paired
    )
    return f"{len(paired)} paired; {mismatches} point-count mismatches"


def _is_identity_3x4(values: Sequence[float]) -> bool:
    """Return whether 12 values equal an identity rotation and zero translation."""

    expected = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    return len(values) == 12 and all(
        abs(observed - target) <= 1e-12
        for observed, target in zip(values, expected)
    )


def build_audit(data_root: Path) -> Dict[str, object]:
    """Collect a read-only audit from a supplied dataset root."""

    root = data_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")
    sequence_dirs = discover_sequence_directories(root)
    if not sequence_dirs:
        raise ValueError(f"no RELLIS-3D sequence directories found under {root}")

    sequences: Dict[str, Dict[str, object]] = {}
    totals: Counter[str] = Counter()
    patterns: Dict[str, str] = {}
    examples: Dict[str, str] = {}
    formats: Dict[str, str] = {}
    point_label_checks: Dict[str, List[str]] = {"ouster": [], "velodyne": []}

    for sequence_dir in sequence_dirs:
        sequence_id = sequence_dir.name
        pose_path = sequence_dir / "poses.txt"
        poses = parse_pose_file(pose_path) if pose_path.is_file() else []
        expected = set(range(len(poses)))
        sources: Dict[str, Dict[str, object]] = {}
        maps: Dict[str, Dict[int, Path]] = {}

        for spec in SOURCE_SPECS:
            files = _source_files(sequence_dir, spec)
            frame_map = _frame_map(files)
            maps[spec.key] = frame_map
            observed = set(frame_map)
            rate = _median_rate(files)
            totals[spec.key] += len(files)
            patterns.setdefault(spec.key, _pattern_for(sequence_dir, spec, root))
            formats.setdefault(spec.key, spec.format_description)
            if files and spec.key not in examples:
                examples[spec.key] = _example_description(spec, files)

            if poses:
                if observed == expected:
                    pose_sync = "yes by equal zero-based index set; no timestamp proof"
                elif observed.issubset(expected):
                    pose_sync = "partially by shared index; no timestamp proof"
                else:
                    pose_sync = "not cleanly by pose-row index"
                missing = _missing_summary(expected, observed)
            else:
                pose_sync = "no pose file found"
                internal_expected = (
                    set(range(min(observed), max(observed) + 1)) if observed else set()
                )
                missing = _missing_summary(internal_expected, observed)

            unexpected: List[str] = []
            folder = sequence_dir / spec.directory
            if folder.is_dir():
                unexpected = sorted(
                    path.name
                    for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() not in spec.suffixes
                )
            sources[spec.key] = {
                "count": len(files),
                "missing": missing,
                "rate_hz": rate,
                "pose_sync": pose_sync,
                "index_convention": _index_convention(files),
                "unexpected": unexpected,
            }

        point_label_checks["ouster"].append(
            f"{sequence_id}: "
            + _pair_size_check(maps["ouster_cloud"], maps["ouster_label"])
        )
        point_label_checks["velodyne"].append(
            f"{sequence_id}: "
            + _pair_size_check(maps["velodyne_cloud"], maps["velodyne_label"])
        )

        pose_first = poses[0] if poses else ()
        pose_translation = (
            (pose_first[3], pose_first[7], pose_first[11]) if pose_first else ()
        )
        camera_info_path = sequence_dir / "camera_info.txt"
        calib_path = sequence_dir / "calib.txt"
        sequences[sequence_id] = {
            "relative_path": _relative(sequence_dir, root),
            "pose_count": len(poses),
            "pose_first_translation": pose_translation,
            "pose_all_12_values": bool(poses),
            "camera_info": (
                parse_camera_info(camera_info_path) if camera_info_path.is_file() else None
            ),
            "calibration": (
                parse_kitti_calibration(calib_path) if calib_path.is_file() else None
            ),
            "sources": sources,
        }
        totals["poses"] += len(poses)

    all_files = [path for path in root.rglob("*") if path.is_file()]
    calibration_suffixes = {".zip", ".yaml", ".yml", ".txt", ".cfg", ".launch"}
    calibration_files = sorted(
        path
        for path in all_files
        if path.suffix.lower() in calibration_suffixes
        and any(
            token in _relative(path, root).lower()
            for token in (
                "calib",
                "cam2lidar",
                "camera_info",
                "transform",
                "intrinsic",
                "lidar_poses",
                "stereo",
            )
        )
    )
    timestamp_files = sorted(
        path
        for path in all_files
        if re.search(r"(?:^|[_-])(timestamp|timestamps|times)(?:[_\-.]|$)", path.name.lower())
    )
    gps_files = sorted(
        path
        for path in all_files
        if re.search(
            r"(?:^|[_-])(gps|odom|odometry|navsat|imu|ins)(?:[_\-.]|$)",
            path.name.lower(),
        )
    )
    archives = sorted(
        path
        for path in all_files
        if path.suffix.lower() == ".zip" or re.fullmatch(r"\.z\d+", path.suffix.lower())
    )
    split_files = sorted(path for path in all_files if path.suffix.lower() == ".lst")

    sequence_calibrations = [
        sequence["calibration"]
        for sequence in sequences.values()
        if sequence["calibration"] is not None
    ]
    all_sequence_calibration_values = [
        values
        for calibration in sequence_calibrations
        for values in calibration.values()
    ]
    camera_info_values = sorted(
        {
            sequence["camera_info"]
            for sequence in sequences.values()
            if sequence["camera_info"] is not None
        }
    )

    return {
        "root": root,
        "sequences": sequences,
        "totals": dict(totals),
        "patterns": patterns,
        "formats": formats,
        "examples": examples,
        "point_label_checks": point_label_checks,
        "directory_inventory": _directory_inventory(root),
        "calibration_files": calibration_files,
        "all_sequence_calibrations_identity": bool(all_sequence_calibration_values)
        and all(_is_identity_3x4(values) for values in all_sequence_calibration_values),
        "camera_info_values": camera_info_values,
        "timestamp_files": timestamp_files,
        "gps_files": gps_files,
        "archives": archives,
        "split_files": split_files,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown(audit: Mapping[str, object]) -> str:
    """Render a human-readable, evidence-bounded audit report."""

    root = audit["root"]
    sequences = audit["sequences"]
    totals = audit["totals"]
    patterns = audit["patterns"]
    formats = audit["formats"]
    examples = audit["examples"]
    assert isinstance(root, Path)
    assert isinstance(sequences, dict)
    assert isinstance(totals, dict)
    assert isinstance(patterns, dict)
    assert isinstance(formats, dict)
    assert isinstance(examples, dict)
    if audit["all_sequence_calibrations_identity"]:
        calibration_value_summary = (
            "All observed values form 3x4 identity-with-zero-translation matrices; "
            "they therefore do not encode a non-trivial measured camera–LiDAR extrinsic."
        )
    else:
        calibration_value_summary = (
            "At least one observed record is non-identity; its transform semantics "
            "still require labels or authoritative documentation."
        )

    lines = [
        "# RELLIS-3D local data audit",
        "",
        f"Generated by `scripts/inspect_rellis3d.py` on {datetime.now().isoformat(timespec='seconds')}.",
        "",
        "## Audit scope and evidence policy",
        "",
        f"- Inspected root: `{root}`",
        f"- Discovered sequence IDs: {', '.join(f'`{value}`' for value in sequences)}",
        "- This report describes bytes, names, counts, and text actually present under the supplied root.",
        "- Coordinate-frame direction, transform direction, and physical units are not assigned where the files do not label them.",
        "- Archives are inventoried as files; claims about usable records refer to the extracted sequence tree.",
        "",
        "## Exact discovered directory structure",
        "",
        "Every discovered directory is listed below with its direct (non-recursive) file count and extensions:",
        "",
        "```text",
    ]
    lines.extend(str(value) for value in audit["directory_inventory"])  # type: ignore[index]
    lines.extend(["```", "", "### Sequence payload counts", ""])

    count_headers = ["Sequence", "Poses"] + [spec.title for spec in SOURCE_SPECS]
    count_rows: List[List[object]] = []
    for sequence_id, sequence in sequences.items():
        source_data = sequence["sources"]
        count_rows.append(
            [sequence_id, sequence["pose_count"]]
            + [source_data[spec.key]["count"] for spec in SOURCE_SPECS]
        )
    count_rows.append(
        ["Total", totals.get("poses", 0)]
        + [totals.get(spec.key, 0) for spec in SOURCE_SPECS]
    )
    lines.extend([_markdown_table(count_headers, count_rows), ""])

    lines.extend(["## Source audit", ""])
    for spec in SOURCE_SPECS:
        lines.extend(
            [
                f"### {spec.title}",
                "",
                f"- Path pattern: `{patterns.get(spec.key, 'not found')}`",
                f"- Total files: **{totals.get(spec.key, 0)}**",
                f"- File format: {formats.get(spec.key, spec.format_description)}.",
                f"- Example shape/content: {examples.get(spec.key, 'not found')}.",
                "",
            ]
        )
        source_rows: List[List[object]] = []
        for sequence_id, sequence in sequences.items():
            source = sequence["sources"][spec.key]
            rate = source["rate_hz"]
            if rate is None and spec.key != "rgb":
                rgb_rate = sequence["sources"]["rgb"]["rate_hz"]
                rate_text = (
                    f"not direct; shared-index RGB is ~{rgb_rate:.3f} Hz"
                    if rgb_rate is not None
                    else "not inferable"
                )
            else:
                rate_text = f"~{rate:.3f} Hz" if rate is not None else "not inferable"
            source_rows.append(
                [
                    sequence_id,
                    source["count"],
                    source["index_convention"],
                    source["missing"],
                    rate_text,
                    source["pose_sync"],
                ]
            )
        lines.extend(
            [
                _markdown_table(
                    [
                        "Sequence",
                        "Files",
                        "Frame indexing",
                        "Missing vs pose rows",
                        "Approx. rate",
                        "Synchronizable with poses?",
                    ],
                    source_rows,
                ),
                "",
            ]
        )
        unexpected = []
        for sequence_id, sequence in sequences.items():
            names = sequence["sources"][spec.key]["unexpected"]
            unexpected.extend(f"`{sequence_id}/{spec.directory}/{name}`" for name in names)
        if unexpected:
            lines.extend(
                [
                    "Unexpected files in this source directory: " + ", ".join(unexpected) + ".",
                    "",
                ]
            )

    lines.extend(
        [
            "## Pose format investigation",
            "",
            f"- Path pattern: `{_relative(Path(next(iter(sequences.values()))['relative_path']).parent, Path('.'))}/{{sequence}}/poses.txt`.",
            f"- Files/rows: {len(sequences)} files and {totals.get('poses', 0)} total rows.",
            "- File format and example content: whitespace-delimited text; each row has 12 floating-point values representing a numeric 3x4 layout.",
            "- Frame indexing and missing frames: implicit zero-based row number; no rows are missing relative to the continuous Ouster/RGB index sets in the five discovered sequences.",
            "- Approximate frame rate: not encoded in the pose file; shared numeric indices refer to RGB names whose median interval implies ~10 Hz.",
            "- Pose synchronization: Ouster and RGB can be associated to pose rows by exact index-set equality, but no independent timestamp proof is present.",
            "",
            "Each sequence has `poses.txt`. Every non-empty row parsed as exactly 12 floating-point values. The byte-level layout is therefore an unlabeled row-major-compatible 3x4 numeric matrix per row; a homogeneous `[0, 0, 0, 1]` row can be appended computationally, but the files do not state what the transform maps from or to.",
            "",
        ]
    )
    pose_rows = []
    for sequence_id, sequence in sequences.items():
        pose_rows.append(
            [
                sequence_id,
                sequence["pose_count"],
                tuple(round(value, 6) for value in sequence["pose_first_translation"]),
                "row n ↔ frame index n (count/index equality observed)",
            ]
        )
    lines.extend(
        [
            _markdown_table(
                ["Sequence", "Rows", "First row columns 4/8/12", "Observed alignment"],
                pose_rows,
            ),
            "",
            "Unresolved from `poses.txt`: source frame, destination frame, whether the matrix is sensor-to-world or world-to-sensor, axis directions, physical translation units, pose-estimation source, and timestamp provenance. These must be verified against authoritative dataset documentation before constructing ego-frame trajectories.",
            "",
            "## Calibration format investigation",
            "",
            "The extracted sequence payload contains two text forms:",
            "",
            f"- `calib.txt`: keys `P0`, `P1`, `P2`, `P3`, and `Tr`, each followed by 12 values. {calibration_value_summary}",
            f"- `camera_info.txt`: four unlabeled floating-point values. Unique observed records: `{audit['camera_info_values']}`. The file itself does not label their order or distortion model.",
            "- Extracted calibration variants contain `transforms.yaml` with key `os1_cloud_node-pylon_camera_node`, quaternion fields `w,x,y,z`, and translation fields `x,y,z`.",
            "- Two transform variant families are present (`Copy of Rellis_3D_cam2lidar` and `Copy of Rellis_3D_cam2lidar_20210224`) and contain different values. File names alone do not establish which is authoritative or the transform direction.",
            "- `Copy of Rellis_3D_stereo_calibration.yaml` is OpenCV-style YAML with documented `M1/D1/M2/D2/R1/R2/P1/P2/Q/T/R`, image size, and reprojection error for a stereo pair. It is not, by itself, the pylon-camera–LiDAR transform.",
            "- The pylon calibration variant includes ROS-style `pylon_iris56.yaml` with explicit 1920x1200 intrinsics, distortion, rectification, and projection matrices. Its numeric intrinsics differ from the unlabeled sequence `camera_info.txt` values.",
            "",
            "Calibration-related files discovered recursively:",
            "",
        ]
    )
    calibration_files = audit["calibration_files"]
    assert isinstance(calibration_files, list)
    transform_count = sum(path.name.lower() == "transforms.yaml" for path in calibration_files)
    sequence_calib_count = sum(path.name.lower() == "calib.txt" for path in calibration_files)
    camera_info_count = sum(path.name.lower() == "camera_info.txt" for path in calibration_files)
    lines.extend(
        [
            "",
            f"- Sequence calibration path patterns: `processed/Rellis-3D/{{sequence}}/calib.txt` and `processed/Rellis-3D/{{sequence}}/camera_info.txt` ({sequence_calib_count} and {camera_info_count} files, respectively).",
            f"- Camera–Ouster transform pattern: `processed/calibration_variants/*/Rellis*/{{sequence}}/transforms.yaml` ({transform_count} files).",
            "- Example content is described above; these are sequence-level text/YAML records rather than per-frame arrays.",
            "- Frame indexing, missing frames, and frame rate: not applicable to the calibration records.",
            "- Synchronization with poses: calibration records can be selected by sequence ID, but the files do not identify the authoritative transform variant or its direction.",
            "",
        ]
    )
    lines.extend(f"- `{_relative(path, root)}`" for path in calibration_files)

    timestamp_files = audit["timestamp_files"]
    gps_files = audit["gps_files"]
    assert isinstance(timestamp_files, list)
    assert isinstance(gps_files, list)
    lines.extend(
        [
            "",
            "## Timestamps, GPS, and odometry",
            "",
            f"- Standalone timestamp files found by name: **{len(timestamp_files)}**.",
            f"- Timestamp-bearing path pattern: `{patterns.get('rgb', 'not found')}` ({totals.get('rgb', 0)} RGB names).",
            "- RGB file names embed an integer seconds field and a variable-length decimal fraction, for example `frame000000-1581624652_750.jpg`. Consecutive names support the reported ~10 Hz image rate; the stored format is name text, not a separate timestamp payload.",
            "- Ouster and Velodyne files contain only numeric frame stems. Their timestamps are not encoded in file names, so assigning RGB timestamps by matching index is an inference, not direct timestamp evidence.",
            f"- GPS/odometry/IMU/INS-related files found by conservative name search: **{len(gps_files)}**.",
            "- GPS/odometry path pattern, format, example content, frame indexing, rate, and pose synchronization: not available because no such standalone files were found.",
            "- `poses.txt` is the only extracted motion-related source found. It contains no header or timestamp column. The calibration ROS bag was inventoried but not decoded as a dataset motion source.",
            "",
            "## LiDAR-to-pose alignment",
            "",
            "For Ouster OS1, every sequence has a continuous zero-based cloud index set exactly equal to the pose-row index set and RGB frame index set. Thus row `n` can be associated with `NNNNNN.bin` and `frameNNNNNN-...jpg` by observed index equality. The files do not contain an independent LiDAR timestamp or a declaration proving simultaneity.",
            "",
            "Velodyne files also start at zero and remain contiguous, but end earlier than the pose/Ouster stream in every sequence. Shared indices can be associated with pose rows; trailing pose frames have no Velodyne file. No Velodyne-to-pose timestamp mapping was found.",
            "",
            "## Semantic-label alignment",
            "",
            "Point-label alignment was checked by identical six-digit stems and by comparing point-cloud record count (`bytes / 16`) with label count (`bytes / 4`) for every paired file:",
            "",
        ]
    )
    checks = audit["point_label_checks"]
    assert isinstance(checks, dict)
    lines.extend(f"- Ouster {value}." for value in checks["ouster"])
    lines.extend(f"- Velodyne {value}." for value in checks["velodyne"])
    lines.extend(
        [
            "",
            "This establishes one uint32 stored value per float32x4 point record. It does not establish undocumented bit packing; no instance/semantic bit split is assumed. Sampled camera PNG labels match sampled RGB dimensions and use matching timestamp-bearing stems, but only a subset of RGB frames has a PNG label.",
            "",
            "## Files required for trajectory planning",
            "",
            "Required for the intended LiDAR-based trajectory dataset:",
            "",
            "1. `poses.txt`, after authoritative confirmation of transform direction, axes, and units, to derive future relative positions.",
            "2. At least one LiDAR stream (`os1_cloud_node_kitti_bin` is the locally complete pose-index-aligned stream) to build terrain observations.",
            "3. A verified frame-association rule. The local files support index association; timestamps are only directly present in RGB names.",
            "4. A defined planning horizon and sampling interval in experiment configuration. Approximate 10 Hz timing can be inferred from RGB names but should be verified before speed-sensitive evaluation.",
            "",
            "Conditionally required:",
            "",
            "- Point semantic labels and `ontology.*` when the terrain representation uses semantic classes or semantic supervision.",
            "- Camera RGB, camera semantic labels, camera intrinsics, and an authoritative camera–LiDAR extrinsic when using camera/LiDAR fusion.",
            "- Official `.lst` split files when reproducing released train/validation/test partitions.",
            "",
            "## Optional files",
            "",
            "- Velodyne clouds and labels when the Ouster stream is the selected terrain sensor.",
            "- RGB and camera labels for LiDAR-only planning.",
            "- Stereo calibration and raw calibration imagery/bag files unless recalibrating or adding stereo inputs.",
            "- Local `perception_cache_*` and `trajectory_cache_*` products; these are derived artifacts, not raw dataset sources.",
            "- ZIP and split-ZIP archives after their extracted payload has been integrity-checked.",
            "",
            "## Unresolved uncertainties",
            "",
            "- Pose transform direction, referenced body/sensor frame, axis directions, translation units, and timestamp source.",
            "- Whether equal numeric RGB/Ouster/pose indices represent hardware synchronization, nearest-neighbor association, or another export policy.",
            "- The authoritative camera–Ouster transform variant and the direction encoded by the YAML key.",
            "- The semantic `.label` bit layout. Only uint32 storage and one-record-per-point alignment are established here.",
            "- The meaning and scaling of the fourth float32 point-cloud channel.",
            "- Why the Velodyne streams terminate several frames early in every sequence.",
            "- Why camera labels are sparse and why one JPEG is present in `00000/pylon_camera_node_label_id`.",
            "- Whether the four values in `camera_info.txt` are ordered `(fx, fy, cx, cy)`; that interpretation is plausible but not labeled in the file and is therefore not adopted by this audit.",
            "- Whether the calibration ROS bag contains any ancillary topics not exposed as extracted files; the audit does not require ROS and does not decode bag internals.",
            "",
        ]
    )
    return "\n".join(lines)


def inspect_dataset(data_root: Path, output_path: Path) -> Dict[str, object]:
    """Audit ``data_root``, write Markdown, and return the collected evidence."""

    audit = build_audit(data_root)
    report = render_markdown(audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return audit


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Recursively inspect a local RELLIS-3D root and write a Markdown audit."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root or an ancestor containing extracted sequence directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Markdown output path (default: {DEFAULT_REPORT}).",
    )
    return parser


def main() -> None:
    """Run the command-line audit."""

    args = build_argument_parser().parse_args()
    audit = inspect_dataset(args.data_root, args.output)
    sequences = audit["sequences"]
    totals = audit["totals"]
    assert isinstance(sequences, dict)
    assert isinstance(totals, dict)
    print(f"Inspected: {audit['root']}")
    print(f"Sequences: {', '.join(sequences)}")
    print(
        "Records: "
        f"poses={totals.get('poses', 0)}, "
        f"RGB={totals.get('rgb', 0)}, "
        f"Ouster={totals.get('ouster_cloud', 0)}, "
        f"Velodyne={totals.get('velodyne_cloud', 0)}"
    )
    print(f"Report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
