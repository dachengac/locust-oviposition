"""Regression for clipped start controls on small screens and Windows display scaling."""
import json
import sys
import time
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from behaviorlab.desktop import BehaviorApp
from behaviorlab.modelpack import create_pack, install_pack

work = ROOT / "build" / "layout-verification"
work.mkdir(parents=True, exist_ok=True)
archive = work / f"layout-{time.time_ns()}.behavior.zip"
create_pack(ROOT / "tests" / "fixtures" / "detector.onnx", archive,
            name="Synthetic layout test", labels=["locust"], input_size=(32, 32))
pack = install_pack(archive, work / "models")
reports = []


def bounds(widget):
    return (widget.winfo_rootx(), widget.winfo_rooty(),
            widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())


def assert_visible(widget, container):
    x0, y0, x1, y1 = bounds(widget)
    a0, b0, a1, b1 = bounds(container)
    assert widget.winfo_ismapped() and x1 - x0 > 10 and y1 - y0 > 10, (widget, "not mapped")
    assert a0 <= x0 < x1 <= a1 and b0 <= y0 < y1 <= b1, (widget, bounds(widget), bounds(container))


for percent in (100, 125, 150):
    for width, height in ((900, 560), (1000, 600), (1180, 640)):
        root = tk.Tk()
        root.attributes("-alpha", 0)
        root.tk.call("tk", "scaling", (4 / 3) * percent / 100)
        app = BehaviorApp(root, ROOT / "build" / "layout-verification", [pack])
        root.geometry(f"{width}x{height}")
        for index in range(30):
            app.video_list.insert("end", f"视频 {index + 1}.mp4 · 待分析")
        root.update()
        assert (root.winfo_width(), root.winfo_height()) == (width, height)
        for widget in (app.start_button, app.stop_button, app.export_button, app.progress):
            assert_visible(widget, root)
            assert_visible(widget, widget.master)
        # The advanced settings are reachable via the sidebar scrollbar even at high DPI.
        app.sidebar_canvas.yview_moveto(1.)
        root.update()
        assert_visible(app.advanced_button, app.sidebar_canvas)
        assert root.bind("<F5>")
        reports.append(dict(display_scale_percent=percent, window=[width, height],
                            start_button=list(bounds(app.start_button)),
                            sidebar_scrollbar_reaches_settings=True))
        for timer in root.tk.call("after", "info"):
            root.after_cancel(timer)
        root.destroy()

report = dict(ok=True, synthetic_model=True, scenarios=reports, checks=["start-visible", "stop-visible", "export-visible", "progress-visible", "settings-scrollable", "F5-bound"])
(ROOT / "assets" / "verification" / "layout-0.1.1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
