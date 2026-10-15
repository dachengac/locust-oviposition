"""Exercise our own Tk app through its callbacks with a synthetic model/video."""
import json
import sys
import time
import tkinter as tk
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from behaviorlab.desktop import BehaviorApp
from behaviorlab.modelpack import create_pack, install_pack
from behaviorlab.project import load_project

work = ROOT / "build" / "ui-integration"
work.mkdir(parents=True, exist_ok=True)
video = work / "synthetic.avi"
writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
assert writer.isOpened()
for i in range(30):
    writer.write(np.full((48, 64, 3), 255 if i < 10 or i >= 20 else 0, np.uint8))
writer.release()
archive = work / f"synthetic-{time.time_ns()}.behavior.zip"
create_pack(ROOT / "tests" / "fixtures" / "detector.onnx", archive, name="Synthetic test only",
            labels=["bright"], input_size=(32, 32), pipeline="detection_presence_v1",
            parameters=dict(min_duration=0, stitch_gap=0))
pack = install_pack(archive, work / "models")
root = tk.Tk()
root.withdraw()
app = BehaviorApp(root, work, [pack])
app.videos = [str(video.resolve())]
app._refresh_videos()
app.video_list.selection_set(0)
app._select_video()
app.start_button.invoke()
deadline = time.monotonic() + 30
while app.running and time.monotonic() < deadline:
    root.update()
    time.sleep(.01)
assert not app.running, "Analysis did not finish"
assert len(app.event_table.get_children()) == 2
app.event_table.selection_set("0")
app.review("已确认")
app.seek(2)
assert app.position == 2
project_path = work / "project.behavior.json"
csv_path = work / "results.csv"
with patch("behaviorlab.desktop.filedialog.asksaveasfilename", return_value=str(project_path)):
    assert app.save()
with patch("behaviorlab.desktop.filedialog.asksaveasfilename", return_value=str(csv_path)):
    app.export()
with patch("behaviorlab.desktop.filedialog.askopenfilename", return_value=str(project_path)):
    app.open_project()
assert app._selected_event() is None
assert load_project(project_path)["results"][0]["events"][0]["review"] == "已确认"
assert csv_path.exists()
root.update()
app.dirty = False
app.close()
report = dict(ok=True, checks=["video-preview", "background-batch-analysis", "timeline", "event-table",
                               "review", "project-save-open", "csv-export"], synthetic_input=True)
(ROOT / "assets" / "verification" / "desktop-integration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
