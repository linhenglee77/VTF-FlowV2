"""Basic parsing tests for the read-only RELLIS-3D audit tool."""

from pathlib import Path
import struct
import tempfile
import unittest

from TerraFlow.scripts.inspect_rellis3d import (
    inspect_image,
    inspect_point_cloud,
    inspect_semantic_label,
    parse_camera_info,
    parse_frame_filename,
    parse_kitti_calibration,
    parse_pose_file,
)


class RellisParserTest(unittest.TestCase):
    """Verify parsers against small deterministic files."""

    def test_parse_camera_and_numeric_frame_names(self) -> None:
        camera = parse_frame_filename("frame000042-1581624652_750.jpg")
        numeric = parse_frame_filename("000042.bin")
        self.assertIsNotNone(camera)
        self.assertIsNotNone(numeric)
        assert camera is not None
        assert numeric is not None
        self.assertEqual(camera.index, 42)
        self.assertAlmostEqual(camera.timestamp_s or 0.0, 1581624652.750)
        self.assertEqual(numeric.index, 42)
        self.assertIsNone(numeric.timestamp_s)
        self.assertIsNone(parse_frame_filename("README.txt"))

    def test_parse_pose_and_calibration_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pose_path = root / "poses.txt"
            pose_path.write_text(
                "1 0 0 1 0 1 0 2 0 0 1 3\n"
                "1 0 0 4 0 1 0 5 0 0 1 6\n",
                encoding="utf-8",
            )
            calib_path = root / "calib.txt"
            calib_path.write_text(
                "Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8"
            )
            camera_path = root / "camera_info.txt"
            camera_path.write_text("10 11 12 13\n", encoding="utf-8")

            poses = parse_pose_file(pose_path)
            calibration = parse_kitti_calibration(calib_path)
            self.assertEqual(len(poses), 2)
            self.assertEqual(poses[0][3], 1.0)
            self.assertEqual(len(calibration["Tr"]), 12)
            self.assertEqual(parse_camera_info(camera_path), (10.0, 11.0, 12.0, 13.0))

    def test_pose_parser_rejects_wrong_column_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            pose_path = Path(temp_dir) / "poses.txt"
            pose_path.write_text("1 0 0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 12"):
                parse_pose_file(pose_path)

    def test_parse_point_and_label_binary_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cloud_path = root / "000000.bin"
            cloud_path.write_bytes(
                struct.pack("<8f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
            )
            label_path = root / "000000.label"
            label_path.write_bytes(struct.pack("<2I", 19, 33))

            cloud = inspect_point_cloud(cloud_path)
            labels = inspect_semantic_label(label_path)
            self.assertEqual((cloud.count, cloud.width, cloud.dtype), (2, 4, "float32"))
            self.assertEqual(cloud.first_values, (1.0, 2.0, 3.0, 4.0))
            self.assertEqual((labels.count, labels.width, labels.dtype), (2, 1, "uint32"))
            self.assertEqual(labels.first_values, (19.0, 33.0))

    def test_parse_png_and_jpeg_headers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            png_path = root / "label.png"
            png_path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", 1920, 1200)
                + bytes((8, 0))
            )
            jpeg_path = root / "image.jpg"
            jpeg_path.write_bytes(
                b"\xff\xd8\xff\xc0"
                + struct.pack(">H", 8)
                + struct.pack(">BHHB", 8, 1200, 1920, 3)
                + b"\xff\xd9"
            )

            png = inspect_image(png_path)
            jpeg = inspect_image(jpeg_path)
            self.assertEqual((png.format, png.height, png.width, png.channels), ("PNG", 1200, 1920, 1))
            self.assertEqual((jpeg.format, jpeg.height, jpeg.width, jpeg.channels), ("JPEG", 1200, 1920, 3))


if __name__ == "__main__":
    unittest.main()

