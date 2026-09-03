"""Inspect TF transforms stored in a ROS1 bag without requiring ROS.

The parser intentionally supports only the ROSBAG v2 records and
``tf2_msgs/TFMessage`` payload needed for the local RELLIS-3D calibration
audit. It never modifies the bag.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import BinaryIO, Iterator


@dataclass(frozen=True)
class TransformRecord:
    """One decoded ROS ``geometry_msgs/TransformStamped`` record."""

    topic: str
    source: str
    parent: str
    child: str
    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    stamp_ns: int


def _read_exact(stream: BinaryIO, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise EOFError(f"expected {count} bytes, received {len(value)}")
    return value


def _u32(value: bytes) -> int:
    return struct.unpack("<I", value)[0]


def _parse_header(value: bytes) -> dict[str, bytes]:
    fields: dict[str, bytes] = {}
    offset = 0
    while offset < len(value):
        size = _u32(value[offset : offset + 4])
        offset += 4
        field = value[offset : offset + size]
        offset += size
        key, content = field.split(b"=", 1)
        fields[key.decode("ascii")] = content
    return fields


def _records(stream: BinaryIO, limit: int | None = None) -> Iterator[tuple[dict[str, bytes], bytes]]:
    emitted = 0
    while limit is None or stream.tell() < limit:
        length = stream.read(4)
        if not length:
            return
        header = _parse_header(_read_exact(stream, _u32(length)))
        data = _read_exact(stream, _u32(_read_exact(stream, 4)))
        yield header, data
        emitted += 1


def _read_string(data: bytes, offset: int) -> tuple[str, int]:
    size = _u32(data[offset : offset + 4])
    offset += 4
    return data[offset : offset + size].decode("utf-8"), offset + size


def decode_tf_message(data: bytes, topic: str, source: str = "unknown") -> list[TransformRecord]:
    """Decode a serialized ``tf2_msgs/TFMessage`` payload."""

    count = _u32(data[:4])
    offset = 4
    transforms: list[TransformRecord] = []
    for _ in range(count):
        _sequence, seconds, nanoseconds = struct.unpack_from("<III", data, offset)
        offset += 12
        parent, offset = _read_string(data, offset)
        child, offset = _read_string(data, offset)
        values = struct.unpack_from("<7d", data, offset)
        offset += 56
        transforms.append(
            TransformRecord(
                topic=topic,
                source=source,
                parent=parent,
                child=child,
                translation=tuple(values[:3]),
                quaternion_xyzw=tuple(values[3:]),
                stamp_ns=seconds * 1_000_000_000 + nanoseconds,
            )
        )
    if offset != len(data):
        raise ValueError(f"TF payload has {len(data) - offset} trailing bytes")
    return transforms


def inspect_tf_bag(path: Path, maximum_chunks: int | None = None) -> list[TransformRecord]:
    """Return all TF records from up to ``maximum_chunks`` uncompressed chunks."""

    connections: dict[int, tuple[str, str]] = {}
    transforms: list[TransformRecord] = []
    chunks = 0
    with path.open("rb") as stream:
        if stream.readline() != b"#ROSBAG V2.0\n":
            raise ValueError("not a ROSBAG v2 file")
        for header, data in _records(stream):
            operation = header.get("op", b"\x00")[0]
            if operation != 0x05:
                continue
            chunks += 1
            compression = header.get("compression", b"").decode("ascii")
            if compression != "none":
                raise ValueError(f"unsupported ROS bag compression: {compression}")
            from io import BytesIO

            for inner_header, inner_data in _records(BytesIO(data), len(data)):
                inner_operation = inner_header.get("op", b"\x00")[0]
                if inner_operation == 0x07:
                    connection = _u32(inner_header["conn"])
                    connection_header = _parse_header(inner_data)
                    connections[connection] = (
                        inner_header["topic"].decode("utf-8"),
                        connection_header.get("callerid", b"unknown").decode("utf-8"),
                    )
                elif inner_operation == 0x02:
                    connection = _u32(inner_header["conn"])
                    topic, source = connections.get(connection, ("", "unknown"))
                    if topic in {"/tf", "/tf_static"}:
                        transforms.extend(decode_tf_message(inner_data, topic, source))
            if maximum_chunks is not None and chunks >= maximum_chunks:
                break
    return transforms


def summarize(transforms: list[TransformRecord]) -> dict[str, object]:
    """Group exact transform values by parent-child edge."""

    grouped: dict[tuple[str, str, str, str], set[tuple[float, ...]]] = defaultdict(set)
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for item in transforms:
        key = (item.topic, item.source, item.parent, item.child)
        grouped[key].add(item.translation + item.quaternion_xyzw)
        counts[key] += 1
    edges = []
    for key in sorted(grouped):
        topic, source, parent, child = key
        values = sorted(grouped[key])
        edges.append(
            {
                "topic": topic,
                "source": source,
                "parent": parent,
                "child": child,
                "message_count": counts[key],
                "unique_value_count": len(values),
                "first_translation_xyz": values[0][:3],
                "first_quaternion_xyzw": values[0][3:],
            }
        )
    return {"decoded_transform_count": len(transforms), "edges": edges}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--maximum-chunks", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(inspect_tf_bag(args.bag, args.maximum_chunks))
    rendered = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
