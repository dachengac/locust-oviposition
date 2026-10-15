import csv
import itertools
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from behaviorlab.analysis import oviposition_events
from behaviorlab.engine import analyze, extract, cache_identity, YoloOnnx, FrameClassifierOnnx
from behaviorlab.modelpack import create_pack, install_pack, load_pack, DEFAULTS, validate_params
from behaviorlab.project import save_project, load_project, export_csv, parse_time
from behaviorlab.vendor.matching import hungarian, linear_assignment
from behaviorlab.vendor.byte_tracker import BYTETracker
from scripts import detect_oviposition as original


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fake_pack(self, pipeline="oviposition_v1", labels=None):
        weights = self.root / "source.onnx"
        weights.write_bytes(b"test-model-data")
        archive = create_pack(weights, self.root / "test.behavior.zip", labels=labels or ["locust"], pipeline=pipeline)
        return install_pack(archive, self.root / "models")

    def make_video(self, name="录像.avi"):
        path = self.root / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(30):
            writer.write(np.full((48, 64, 3), 255 if index < 10 or index >= 20 else 0, np.uint8))
        writer.release()
        return path

    def test_pack_roundtrip_and_duplicate_import(self):
        pack = self.fake_pack()
        again = install_pack(self.root / "test.behavior.zip", self.root / "models")
        self.assertEqual(pack.identity, again.identity)
        self.assertEqual(pack.root, again.root)
        self.assertEqual(pack.params, DEFAULTS)

    def test_tampered_weights_rejected(self):
        pack = self.fake_pack()
        pack.weights.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "校验失败"):
            load_pack(pack.root)

    def test_unsafe_archives_rejected_without_writes(self):
        for index, filename in enumerate(["../outside.txt", "/absolute.txt", "C:/file.txt", "folder\\..\\escape.txt"]):
            archive = self.root / f"bad{index}.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr(filename, "unsafe")
            with self.assertRaises(ValueError):
                install_pack(archive, self.root / "models")
        self.assertFalse((self.root / "outside.txt").exists())

    def test_symlink_archive_rejected(self):
        archive = self.root / "symlink.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            item = zipfile.ZipInfo("model.onnx")
            item.external_attr = 0o120777 << 16
            bundle.writestr(item, "../source.onnx")
        with self.assertRaises(ValueError):
            install_pack(archive, self.root / "models")

    def test_contract_and_numeric_ranges(self):
        for changes in [{"conf": 0}, {"sample_fps": 0}, {"win_sec": float("nan")}, {"stitch_gap": -1}, {"min_track_len": 1.5}, {"other": 1}]:
            with self.assertRaises(ValueError):
                validate_params(changes)
        pack = self.fake_pack()
        manifest_path = pack.root / "manifest.json"
        manifest = pack.manifest.copy()
        manifest["pipeline"] = "classification_v1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不兼容"):
            load_pack(pack.root)

    def test_optimal_assignment_against_exhaustive_search(self):
        random = np.random.default_rng(22)
        for size in range(1, 6):
            for _ in range(5):
                cost = random.random((size, size))
                result = hungarian(cost)
                expected = min(sum(cost[i, j] for i, j in enumerate(p)) for p in itertools.permutations(range(size)))
                self.assertAlmostEqual(sum(cost[i, j] for i, j in enumerate(result)), expected)
        matches, rows, cols = linear_assignment(np.array([[.1, .9], [.9, .95]]), .8)
        self.assertEqual(matches.tolist(), [[0, 0]])
        self.assertEqual(rows, (1,))
        self.assertEqual(cols, (1,))

    def test_tracker_survives_short_missed_detection_and_uses_low_scores(self):
        tracker = BYTETracker(SimpleNamespace(track_thresh=.25, track_buffer=30, match_thresh=.8, mot20=False))
        first = tracker.update(np.array([[10., 10., 30., 30., .9]]), (100, 100), (100, 100))
        tid = first[0].track_id
        second = tracker.update(np.array([[11., 10., 31., 30., .2]]), (100, 100), (100, 100))
        self.assertEqual(second[0].track_id, tid)
        self.assertEqual(tracker.update(np.zeros((0, 5)), (100, 100), (100, 100)), [])
        again = tracker.update(np.array([[12., 10., 32., 30., .9]]), (100, 100), (100, 100))
        self.assertEqual(again[0].track_id, tid)

    def test_temporal_results_match_original_algorithm(self):
        random = np.random.default_rng(7)
        trajectory = []
        for second in range(300):
            x = second * 2 if second < 60 else 120 + random.normal(0, .15) if second < 240 else 120 + (second - 240) * 2
            trajectory.append((second, x, 30., 20., 12.))
        tracks = {1: trajectory}
        params = dict(DEFAULTS, win_sec=20, min_duration=60, frag_min_duration=60, merge_gap=120)
        events = oviposition_events(tracks, params)
        self.assertEqual(len(events), 1)
        args = SimpleNamespace(**params, imgsz=640, cache_tracks=None)
        with patch.object(original, "extract_tracks", return_value=(tracks, 30)):
            expected = original.analyze_video(None, "test.mp4", args)
        self.assertEqual(events[0]["start"], expected["pred_start"])
        self.assertEqual(events[0]["end"], expected["pred_end"])

    def test_cache_changes_for_model_video_or_inference_parameters(self):
        pack = self.fake_pack()
        key = cache_identity(pack, "video1", DEFAULTS)
        self.assertNotEqual(key, cache_identity(pack, "video2", DEFAULTS))
        self.assertNotEqual(key, cache_identity(pack, "video1", dict(DEFAULTS, conf=.5)))
        self.assertNotEqual(key, cache_identity(pack, "video1", dict(DEFAULTS, sample_fps=2)))
        self.assertEqual(key, cache_identity(pack, "video1", dict(DEFAULTS, min_duration=60)))
        changed = pack.manifest.copy()
        changed["name"] = "new model"
        from behaviorlab.modelpack import ModelPack
        self.assertNotEqual(key, cache_identity(ModelPack(pack.root, changed, pack.weights, "other-sha"), "video1", DEFAULTS))

    def test_video_timestamps_cache_cancel_and_unreadable_input(self):
        pack = self.fake_pack(pipeline="detection_presence_v1")
        video = self.make_video()
        class FakeAdapter:
            def __init__(self, ignored):
                pass
            def predict(self, frame, confidence):
                return np.array([[10., 10., 30., 30., .9, 0]]) if frame.mean() > 128 else np.zeros((0, 6))
        params = dict(DEFAULTS, min_duration=0, stitch_gap=0)
        with patch.dict("behaviorlab.engine.ADAPTER_REGISTRY", yolo_onnx=FakeAdapter):
            result = analyze(pack, video, params, self.root / "cache")
            self.assertEqual([(e["start"], e["end"]) for e in result["events"]], [(0., 1.), (2., 3.)])
            again = analyze(pack, video, params, self.root / "cache")
            self.assertTrue(again["cached"])
            cache_path = Path(result["cache_file"])
            cache_path.write_text("{broken", encoding="utf-8")
            self.assertFalse(analyze(pack, video, params, self.root / "cache")["cached"])
            cancel = threading.Event()
            cancel.set()
            with self.assertRaises(InterruptedError):
                analyze(pack, video, params, None, cancel)
            with self.assertRaises(ValueError):
                analyze(pack, self.root / "missing.avi", params)

    def test_review_project_and_csv_preserve_original_failures(self):
        result = dict(video="实验.avi", status="完成", info={"duration": 60}, model_name="飞蝗", model_id="abc", parameters={},
                      events=[dict(start=12., end=32., behavior="产卵", review="已修正", source="自动识别", note="修正起点", original=dict(start=10., end=30., behavior="产卵"))])
        failed = dict(video="损坏.avi", status="出错", error="无法读取", events=[])
        save_project(self.root / "project.json", [result, failed])
        loaded = load_project(self.root / "project.json")["results"]
        export_csv(self.root / "result.csv", loaded)
        with open(self.root / "result.csv", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["起点秒"], "12.0")
        self.assertEqual(rows[0]["原始起点秒"], "10.0")
        self.assertEqual(rows[1]["分析状态"], "出错")
        self.assertEqual(rows[1]["错误"], "无法读取")

    def test_time_parsing(self):
        self.assertEqual(parse_time("01:02:03.5"), 3723.5)
        self.assertEqual(parse_time("30.25"), 30.25)
        for bad in ["nan", "-1", "00:80:00", "abc", "1:2:3:4"]:
            with self.assertRaises(ValueError):
                parse_time(bad)


    # These tiny, deterministic ONNX graphs test the actual shipped inference engine.
    # They are fixtures, not scientific models, and are excluded from distributions.
    def test_real_yolo_and_classifier_onnx_video_pipeline(self):
        fixtures = Path(__file__).parent / "fixtures"
        video = self.make_video()
        for adapter, pipeline, filename, labels in [
                ("yolo_onnx", "detection_presence_v1", "detector.onnx", ["bright"]),
                ("frame_classifier_onnx", "classification_v1", "classifier.onnx", ["bright", "dark"])]:
            archive = self.root / f"{filename}.zip"
            create_pack(fixtures / filename, archive, adapter=adapter, pipeline=pipeline, labels=labels,
                        input_size=(32, 32), parameters=dict(min_duration=0, stitch_gap=0))
            pack = install_pack(archive, self.root / f"模型-{filename}")
            result = analyze(pack, video, pack.params, self.root / "real-cache")
            bright = [(e["start"], e["end"]) for e in result["events"] if e["behavior"] == "bright"]
            self.assertEqual(bright, [(0., 1.), (2., 3.)])
            if adapter == "frame_classifier_onnx":
                self.assertEqual([(e["start"], e["end"]) for e in result["events"] if e["behavior"] == "dark"], [(1., 2.)])


if __name__ == "__main__":
    unittest.main()
