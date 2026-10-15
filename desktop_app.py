"""Windows entry point; training dependencies are never imported here."""
import argparse
import json
import os
import sys
import tkinter as tk
from pathlib import Path

from behaviorlab.desktop import BehaviorApp
from behaviorlab.engine import analyze, YoloOnnx, FrameClassifierOnnx
from behaviorlab.modelpack import install_pack, load_pack
from behaviorlab.project import save_project, export_csv


def app_root():
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent


def available_packs(home):
    packs, errors = [], []
    model_dir = home / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Bundled model folders load in place, avoiding a second 36 MB model copy.
    for manifest in sorted((app_root() / "models").glob("*/manifest.json")):
        try:
            packs.append(load_pack(manifest.parent))
        except Exception as exc:
            errors.append(f"{manifest.parent.name}: {exc}")
    for archive in sorted((app_root() / "models").glob("*.behavior.zip")):
        try:
            install_pack(archive, model_dir)
        except Exception as exc:
            errors.append(f"{archive.name}: {exc}")
    for manifest in sorted(model_dir.glob("*/manifest.json")):
        try:
            pack = load_pack(manifest.parent)
            if not any(existing.identity == pack.identity for existing in packs):
                packs.append(pack)
        except Exception as exc:
            errors.append(f"{manifest.parent.name}: {exc}")
    return packs, errors


def main():
    parser = argparse.ArgumentParser(description="BehaviorLab desktop and batch analysis")
    parser.add_argument("--self-test", metavar="REPORT")
    parser.add_argument("--model", help="已解压的模型包目录或 .behavior.zip")
    parser.add_argument("--analyze", nargs="+", metavar="VIDEO")
    parser.add_argument("--output", default="analysis-output", help="命令行分析输出目录")
    args = parser.parse_args()
    home = Path(os.environ.get("BEHAVIORLAB_HOME", str(app_root() / "data"))).resolve()
    home.mkdir(parents=True, exist_ok=True)
    packs, errors = available_packs(home)
    if args.model:
        pack = install_pack(args.model, home / "models") if Path(args.model).is_file() else load_pack(args.model)
        packs = [pack]
    if args.analyze:
        if not packs:
            raise ValueError("请提供模型包")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        rows = [analyze(packs[-1], video, packs[-1].params, home / "cache") for video in args.analyze]
        save_project(output / "project.behavior.json", rows, packs[-1].root)
        export_csv(output / "results.csv", rows)
        return
    root = tk.Tk()
    app = BehaviorApp(root, home, packs)
    if errors:
        app.status.set("部分模型包未能加载：" + "; ".join(errors))
    if args.self_test:
        import numpy as np
        root.update()
        report = dict(ok=True, model_count=len(packs), errors=errors, window=[root.winfo_width(), root.winfo_height()],
                      preview=[app.preview.winfo_width(), app.preview.winfo_height()],
                      table=[app.event_table.winfo_width(), app.event_table.winfo_height()],
                      training_dependencies_loaded=any(name in sys.modules for name in ["torch", "ultralytics", "gradio", "scipy"]))
        if not packs or errors:
            raise ValueError("默认模型包未通过加载验证")
        pack = packs[-1]
        adapter = YoloOnnx(pack) if pack.manifest["adapter"] == "yolo_onnx" else FrameClassifierOnnx(pack)
        prediction = adapter.predict(np.zeros((480, 854, 3), dtype=np.uint8), pack.params["conf"])
        report["prediction_count"] = len(prediction)
        Path(args.self_test).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        root.destroy()
    else:
        root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        try:
            location = app_root() / "behaviorlab-error.log"
            location.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        if getattr(sys, "frozen", False) and not any(arg in sys.argv for arg in ["--self-test", "--analyze"]):
            from tkinter import messagebox
            messagebox.showerror("BehaviorLab 启动失败", str(exc))
        elif not getattr(sys, "frozen", False):
            traceback.print_exc()
        sys.exit(1)
