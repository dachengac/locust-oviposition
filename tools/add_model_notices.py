"""Add provenance/license notices to the default model ZIP without changing model data."""
import hashlib
import json
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / "models" / "locust-oviposition.behavior.zip"
with zipfile.ZipFile(archive) as source:
    manifest = json.loads(source.read("manifest.json"))
    weights = source.read(manifest["weights"])
    assert hashlib.sha256(weights).hexdigest() == manifest["sha256"]
    contents = {info.filename: source.read(info) for info in source.infolist() if not info.is_dir()}
provenance = (
    "Default locust detector model pack\n\n"
    "Source repository: https://github.com/dachengac/locust-oviposition\n"
    "Source weights: https://github.com/dachengac/locust-oviposition/releases/download/v1.0/best.pt\n"
    "Source PT SHA256: 56e3f750492ff16927142ff01e1545a82b073e1e4435c73e824c0c9475b78de8\n"
    f"ONNX SHA256: {manifest['sha256']}\n"
    "Architecture: single-class YOLO11s, class locust.\n"
    "Export: Ultralytics 8.3.203, CPU PyTorch 2.5.1, ONNX opset 12, FP32, fixed batch 1, 640x640, no embedded NMS.\n"
    "The detector detects locusts. Oviposition is inferred by the declared temporal pipeline.\n"
    "Three synthetic frames were compared numerically after export; desktop behavioral accuracy was not remeasured.\n"
    "SOURCE-LICENSE.txt reproduces the source repository MIT license.\n"
    "EXPORTER-LICENSE.txt reproduces the Ultralytics exporter's AGPL-3.0 license notice;\n"
    "the application license does not replace upstream third-party terms.\n"
)
contents.update({"PROVENANCE.txt": provenance.encode("utf-8"),
                 "SOURCE-LICENSE.txt": (ROOT / "LICENSE").read_bytes(),
                 "EXPORTER-LICENSE.txt": (ROOT / "assets" / "Ultralytics-LICENSE.txt").read_bytes()})
temporary = archive.with_suffix(".tmp.zip")
with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as target:
    for name, data in contents.items():
        target.writestr(name, data)
os.replace(temporary, archive)
print(json.dumps(dict(model_sha256=manifest["sha256"], archive_bytes=archive.stat().st_size,
                      notices=["PROVENANCE.txt", "SOURCE-LICENSE.txt", "EXPORTER-LICENSE.txt"])))
