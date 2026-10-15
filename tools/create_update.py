"""Create a data-preserving update containing the EXE and any changed runtime files."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behaviorlab import __version__

previous = ROOT / "dist" / "BehaviorLab"
previous_archive = ROOT / "dist" / "BehaviorLab-Windows-x64.zip"
current = ROOT / "dist" / f"BehaviorLab-{__version__}"
if not previous_archive.is_file() or not (current / "BehaviorLab.exe").is_file():
    raise ValueError("Both previous and current distributions must exist")


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


selected = [current / "BehaviorLab.exe"]
with zipfile.ZipFile(previous_archive) as old_bundle:
    for source in sorted((current / "_internal").rglob("*")):
        if not source.is_file():
            continue
        old_name = "BehaviorLab/" + source.relative_to(current).as_posix()
        try:
            old_info = old_bundle.getinfo(old_name)
        except KeyError:
            selected.append(source)
            continue
        if old_info.file_size != source.stat().st_size:
            selected.append(source)
            continue
        checksum = hashlib.sha256()
        with old_bundle.open(old_name) as old_stream:
            while chunk := old_stream.read(1024 * 1024):
                checksum.update(chunk)
        if checksum.hexdigest() != digest(source):
            selected.append(source)
for name in ["DESKTOP.md", "VALIDATION.md", "开始使用.txt"]:
    selected.append(current / name)
output = ROOT / "dist" / f"BehaviorLab-{__version__}-Update.zip"
instructions = (
    f"BehaviorLab {__version__} 界面更新\n\n"
    "1. 先关闭正在运行的 BehaviorLab。\n"
    "2. 将此压缩包解压到旧版软件目录，让新的 BehaviorLab.exe 覆盖旧版同名文件。\n"
    "   此目录里原本应有 BehaviorLab.exe、_internal 和 models。\n"
    "3. 重新双击 BehaviorLab.exe，窗口顶部会显示蓝色“开始批量分析”按钮。\n"
    "4. 添加视频后点击该按钮，或按 F5 开始分析。左侧设置可用滚动条查看。\n\n"
    "此更新不包含视频、模型、缓存或项目文件，不会覆盖这些数据。\n"
)
with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
    for source in selected:
        bundle.write(source, source.relative_to(current).as_posix())
    bundle.writestr("更新说明.txt", instructions.encode("utf-8-sig"))
report = dict(version=__version__, file=output.name, bytes=output.stat().st_size,
              sha256=digest(output), files=[p.relative_to(current).as_posix() for p in selected])
(ROOT / "dist" / f"UPDATE_REPORT-{__version__}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
