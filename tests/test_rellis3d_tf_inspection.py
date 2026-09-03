"""Tests for the dependency-free ROS TF payload decoder."""

from __future__ import annotations

import struct
import unittest

from TerraFlow.scripts.inspect_rellis3d_tf import decode_tf_message


def ros_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


class RellisTfInspectionTest(unittest.TestCase):
    def test_decode_single_transform(self) -> None:
        payload = (
            struct.pack("<I", 1)
            + struct.pack("<III", 7, 12, 34)
            + ros_string("base_link")
            + ros_string("ouster1/os1_lidar")
            + struct.pack("<7d", 0.19, 0.0, 0.84836, 0.0, 0.0, 1.0, 0.0)
        )
        result = decode_tf_message(payload, "/tf_static", "/os_cloud_node")
        self.assertEqual(len(result), 1)
        transform = result[0]
        self.assertEqual(transform.parent, "base_link")
        self.assertEqual(transform.child, "ouster1/os1_lidar")
        self.assertEqual(transform.source, "/os_cloud_node")
        self.assertEqual(transform.stamp_ns, 12_000_000_034)
        self.assertEqual(transform.quaternion_xyzw, (0.0, 0.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
