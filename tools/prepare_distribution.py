"""Populate a lean PyInstaller folder and make software-only and full portable ZIPs."""
import argparse
import importlib.metadata
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behaviorlab import __version__
DIST = ROOT / "dist" / f"BehaviorLab-{__version__}"


def populate():
    if not (DIST / "BehaviorLab.exe").is_file():
        raise ValueError("请先运行 PyInstaller")
    (DIST / "models").mkdir(exist_ok=True)
    from behaviorlab.modelpack import install_pack
    for package in (ROOT / "models").glob("*.behavior.zip"):
        installed = install_pack(package, DIST / "models")
        with zipfile.ZipFile(package) as bundle:
            for name in ("PROVENANCE.txt", "SOURCE-LICENSE.txt", "EXPORTER-LICENSE.txt"):
                if name in bundle.namelist():
                    (installed.root / name).write_bytes(bundle.read(name))
        shutil.copy2(package, ROOT / "dist" / package.name)
    if not list((DIST / "models").glob("*/manifest.json")):
        raise ValueError("默认模型包缺失")
    for name in ["LICENSE", "THIRD_PARTY.md"]:
        shutil.copy2(ROOT / name, DIST / name)
    for name in ["DESKTOP.md", "MODEL_PACK.md", "VALIDATION.md"]:
        shutil.copy2(ROOT / "docs" / name, DIST / name)
    notices = DIST / "licenses"
    notices.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "behaviorlab" / "vendor" / "ByteTrack-LICENSE.txt", notices / "ByteTrack-LICENSE.txt")
    if (ROOT / "assets" / "Ultralytics-LICENSE.txt").is_file():
        shutil.copy2(ROOT / "assets" / "Ultralytics-LICENSE.txt", notices / "Ultralytics-LICENSE.txt")
    for package in ["numpy", "opencv-python-headless", "pillow"]:
        metadata = importlib.metadata.distribution(package)
        for item in metadata.files or []:
            if any(token in str(item).lower() for token in ["license", "copying"]):
                source = Path(metadata.locate_file(item))
                if source.is_file():
                    target = notices / package / str(item)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
    base = Path(sys.base_prefix)
    for name, source in [("Python-LICENSE.txt", base / "LICENSE.txt"),
                         ("Tcl-LICENSE.txt", base / "tcl" / "tcl8.6" / "license.terms"),
                         ("Tk-LICENSE.txt", base / "tcl" / "tk8.6" / "license.terms")]:
        if source.is_file():
            shutil.copy2(source, notices / name)
    (DIST / "开始使用.txt").write_text(
        "BehaviorLab · 行为识别工作台\n\n"
        "双击 BehaviorLab.exe 即可，无需安装 Python。\n"
        "保留整个解压目录；不能只复制 EXE。\n"
        "默认飞蝗模型来自 GitHub v1.0 发布附件。\n"
        "添加视频 → 开始批量分析 → 复核时段 → 保存项目 / 导出 CSV。\n"
        "换模型：点击右上方“导入模型包”，选择 .behavior.zip。\n"
        "详细操作看 DESKTOP.md，模型格式看 MODEL_PACK.md。\n"
        "所有分析在本机 CPU 上运行。预览为静音视频。\n"
        "识别精度需在真实实验录像上复核。\n", encoding="utf-8-sig")
    verification = DIST / "verification"
    verification.mkdir(exist_ok=True)
    for source in (ROOT / "assets" / "verification").glob("*.json"):
        shutil.copy2(source, verification / source.name)


def archive():
    entries = sorted(p for p in DIST.rglob("*") if p.is_file() and p.relative_to(DIST).parts[0] != "data")
    model_files = [p for p in entries if p.relative_to(DIST).parts[0] == "models"]
    software_files = [p for p in entries if p not in model_files]
    size_file = DIST / "SIZE_REPORT.json"
    software_bytes = sum(p.stat().st_size for p in software_files if p != size_file)
    model_bytes = sum(p.stat().st_size for p in model_files)
    brief = dict(software_unpacked_bytes=software_bytes, model_unpacked_bytes=model_bytes,
                 note="默认模型直接读取，无需重复解压或安装运行框架；导入其他模型、视频和缓存另占空间。")
    base_software_bytes = software_bytes
    for _ in range(4):
        size_file.write_text(json.dumps(brief, indent=2, ensure_ascii=False), encoding="utf-8")
        total = base_software_bytes + size_file.stat().st_size
        if brief["software_unpacked_bytes"] == total:
            break
        brief["software_unpacked_bytes"] = total
    software_bytes = total
    if size_file not in entries:
        entries.append(size_file)
        software_files.append(size_file)
    archives = {}
    for label, selected in [("Software", software_files), ("Windows-x64", entries)]:
        target = ROOT / "dist" / f"BehaviorLab-{label}-{__version__}.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in selected:
                bundle.write(path, DIST.name + "/" + path.relative_to(DIST).as_posix())
        archives[label] = dict(file=target.name, bytes=target.stat().st_size,
                               sha256=__import__("hashlib").sha256(target.read_bytes()).hexdigest())
    report = dict(software_unpacked_bytes=software_bytes, model_unpacked_bytes=model_bytes,
                  total_initial_unpacked_bytes=software_bytes + model_bytes,
                  standalone_model_package_bytes=(ROOT / "dist" / "locust-oviposition.behavior.zip").stat().st_size,
                  note=brief["note"],
                  archives=archives)
    (ROOT / "dist" / f"SIZE_REPORT-{__version__}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    populate()
    if args.archive:
        archive()
