"""Native Tk desktop UI; inference runs off the UI thread."""
from __future__ import annotations

import bisect
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from . import __version__
from .engine import analyze, video_info
from .modelpack import load_pack, install_pack, create_pack, validate_params
from .project import (format_time, parse_time, validate_interval,
                      save_project, load_project, export_csv)

VIDEO_TYPES = [("实验录像", "*.mp4 *.avi *.mov *.mkv *.m4v"), ("所有文件", "*.*")]


class BehaviorApp:
    def __init__(self, root, home, packs):
        self.root, self.home, self.packs = root, Path(home), packs
        self.videos, self.results = [], {}
        self.cap, self.info, self.current_video = None, None, None
        self.position, self.playing, self.running, self.dirty = 0., False, False, False
        self.preview_tracks = {}
        self.messages, self.cancel = queue.Queue(), threading.Event()
        self.root.title(f"BehaviorLab {__version__} · 行为识别工作台")
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        width, height = min(1220, screen_width - 60), min(850, screen_height - 120)
        self.root.geometry(f"{max(900, width)}x{max(560, height)}")
        self.root.minsize(900, 560)
        self.root.configure(bg="#f1f4f8")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self.model_index = tk.StringVar()
        self.status = tk.StringVar(value="导入实验录像，然后开始分析。所有识别在本机完成。")
        self.time_text = tk.StringVar(value="00:00:00.000 / 00:00:00.000")
        self.params = {"conf": tk.DoubleVar(value=.7), "sample_fps": tk.DoubleVar(value=1),
                       "min_duration": tk.DoubleVar(value=15)}
        self.cache = tk.BooleanVar(value=True)
        self.overlay = tk.BooleanVar(value=True)
        self._layout()
        self._refresh_models()
        self.root.bind("<F5>", lambda event: self.start_analysis())
        self.root.after(100, self._poll)

    def _style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10), background="#f1f4f8", foreground="#1d2b42")
        style.configure("TButton", padding=(10, 5))
        style.configure("Accent.TButton", background="#3454cf", foreground="white")
        style.map("Accent.TButton", background=[("active", "#2442b2"), ("disabled", "#9ca9d3")])
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 19, "bold"))
        style.configure("Hint.TLabel", foreground="#66758c", font=("Microsoft YaHei UI", 9))
        style.configure("Treeview", rowheight=28, background="white", fieldbackground="white")
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"), background="#e8edf7")
        style.map("Treeview", background=[("selected", "#dce5ff")], foreground=[("selected", "#17305f")])

    def _layout(self):
        toolbar = ttk.Frame(self.root, padding=(20, 12))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        toolbar.grid(row=0, column=0, sticky="ew")
        ttk.Label(toolbar, text="BehaviorLab", style="Title.TLabel").pack(side="left")
        for text, callback in [("打开项目", self.open_project), ("保存项目", self.save),
                               ("创建模型包", self.pack_dialog), ("导入模型包", self.import_model)]:
            ttk.Button(toolbar, text=text, command=callback).pack(side="right", padx=4)
        # Reserve space above the resizable body so the primary action never clips.
        analysis_bar = ttk.Frame(self.root, padding=(18, 0, 18, 10))
        analysis_bar.grid(row=1, column=0, sticky="ew")
        self.start_button = ttk.Button(analysis_bar, text="开始批量分析", style="Accent.TButton",
                                       command=self.start_analysis, width=18)
        self.start_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(analysis_bar, text="停止分析", command=self.cancel.set,
                                      state="disabled", width=10)
        self.stop_button.pack(side="left", padx=(0, 12))
        ttk.Label(analysis_bar, text="添加视频后点击开始，也可按 F5", style="Hint.TLabel").pack(side="left")
        body = ttk.Frame(self.root, padding=(18, 0, 18, 6))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        ui_scale = max(1., float(self.root.tk.call("tk", "scaling")) / (4 / 3))
        sidebar = ttk.Frame(body, width=round(274 * min(ui_scale, 1.5)))
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        sidebar.grid_propagate(False)
        sidebar.rowconfigure(0, weight=1)
        sidebar.columnconfigure(0, weight=1)
        self.sidebar_canvas = tk.Canvas(sidebar, height=1, background="#f1f4f8", highlightthickness=0)
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        self.sidebar_scroll = ttk.Scrollbar(sidebar, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_scroll.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=self.sidebar_scroll.set)
        left = ttk.Frame(self.sidebar_canvas)
        sidebar_window = self.sidebar_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", lambda event: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        def resize_sidebar(event):
            self.sidebar_canvas.itemconfigure(sidebar_window, width=event.width)
            self.model_hint.configure(wraplength=max(140, event.width - 32))
        self.sidebar_canvas.bind("<Configure>", resize_sidebar)
        model_frame = ttk.LabelFrame(left, text="01  识别模型", padding=12)
        model_frame.pack(fill="x", pady=(0, 10))
        self.model_combo = ttk.Combobox(model_frame, state="readonly", textvariable=self.model_index)
        self.model_combo.pack(fill="x")
        self.model_combo.bind("<<ComboboxSelected>>", self._model_changed)
        self.model_hint = ttk.Label(model_frame, text="", wraplength=225, style="Hint.TLabel")
        self.model_hint.pack(fill="x", pady=(8, 0))
        videos_frame = ttk.LabelFrame(left, text="02  实验录像", padding=10)
        videos_frame.pack(fill="x", pady=(0, 10))
        buttons = ttk.Frame(videos_frame)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="添加视频", command=self.add_videos).pack(side="left")
        ttk.Button(buttons, text="移除", command=self.remove_video).pack(side="right")
        video_list_frame = ttk.Frame(videos_frame)
        video_list_frame.pack(fill="x", pady=8)
        self.video_list = tk.Listbox(video_list_frame, height=5, font=("Microsoft YaHei UI", 10), relief="flat",
                                    bg="white", selectbackground="#dce5ff", selectforeground="#17305f", exportselection=False)
        video_scroll = ttk.Scrollbar(video_list_frame, orient="vertical", command=self.video_list.yview)
        video_scroll.pack(side="right", fill="y")
        self.video_list.configure(yscrollcommand=video_scroll.set)
        self.video_list.pack(fill="both", expand=True)
        self.video_list.bind("<<ListboxSelect>>", self._select_video)
        options = ttk.LabelFrame(left, text="03  分析设置", padding=12)
        options.pack(fill="x", pady=(0, 10))
        for label, key, low, high, step in [("置信度", "conf", .01, 1, .05), ("每秒抽帧", "sample_fps", .1, 30, .1),
                                           ("最短时段 / 分钟", "min_duration", 0, 120, 1)]:
            row = ttk.Frame(options)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label).pack(side="left")
            ttk.Spinbox(row, textvariable=self.params[key], from_=low, to=high, increment=step, width=7).pack(side="right")
        ttk.Checkbutton(options, text="复用检测与轨迹缓存", variable=self.cache).pack(anchor="w", pady=4)
        self.advanced_button = ttk.Button(options, text="高级分析参数", command=self.advanced)
        self.advanced_button.pack(fill="x")
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=3, minsize=60)
        right.rowconfigure(3, weight=2, minsize=60)
        right.columnconfigure(0, weight=1)
        self.preview = tk.Canvas(right, background="#142034", highlightthickness=0, height=260)
        self.preview.grid(row=0, column=0, sticky="nsew")
        self.preview.create_text(350, 120, text="添加录像开始查看\n支持 MP4 · AVI · MOV · MKV", fill="#9fb0cd", font=("Microsoft YaHei UI", 14), justify="center", tags="placeholder")
        self.preview.bind("<Configure>", lambda event: self.show_frame())
        controls = ttk.Frame(right, padding=(0, 6))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(3, weight=1)
        ttk.Button(controls, text="播放 / 暂停", command=self.toggle_play).grid(row=0, column=0, sticky="w")
        ttk.Button(controls, text="−1 秒", command=lambda: self.seek(self.position - 1)).grid(row=0, column=1, padx=4)
        ttk.Button(controls, text="+1 秒", command=lambda: self.seek(self.position + 1)).grid(row=0, column=2)
        ttk.Checkbutton(controls, text="显示轨迹框", variable=self.overlay, command=self.show_frame).grid(row=0, column=3, sticky="e")
        ttk.Label(controls, textvariable=self.time_text).grid(row=1, column=0, columnspan=4, sticky="w", pady=(3, 0))
        timeline_frame = ttk.Frame(right)
        timeline_frame.grid(row=2, column=0, sticky="ew")
        self.slider = ttk.Scale(timeline_frame, from_=0, to=1, command=self._slider_seek)
        self.slider.pack(fill="x")
        self.timeline = tk.Canvas(timeline_frame, height=56, background="white", highlightthickness=0)
        self.timeline.pack(fill="x", pady=(2, 8))
        self.timeline.bind("<Button-1>", self._timeline_seek)
        self.timeline.bind("<Configure>", lambda event: self.draw_timeline())
        event_frame = ttk.LabelFrame(right, text="行为时段 · 双击事件跳转到起点", padding=8)
        event_frame.grid(row=3, column=0, sticky="nsew")
        columns = ("behavior", "start", "end", "duration", "review")
        self.event_table = ttk.Treeview(event_frame, columns=columns, show="headings", selectmode="browse", height=6)
        for key, label, width in zip(columns, ["行为", "起点", "终点", "时长 / 分钟", "复核状态"], [100, 125, 125, 90, 105]):
            self.event_table.heading(key, text=label)
            self.event_table.column(key, width=width, minwidth=65, stretch=True)
        scroll = ttk.Scrollbar(event_frame, orient="vertical", command=self.event_table.yview)
        horizontal_scroll = ttk.Scrollbar(event_frame, orient="horizontal", command=self.event_table.xview)
        self.event_table.configure(yscrollcommand=scroll.set, xscrollcommand=horizontal_scroll.set)
        horizontal_scroll.pack(side="bottom", fill="x")
        scroll.pack(side="right", fill="y")
        self.event_table.pack(fill="both", expand=True)
        self.event_table.bind("<Double-1>", self._jump_event)
        actions = ttk.Frame(right, padding=(0, 8))
        actions.grid(row=4, column=0, sticky="ew")
        for column, (label, callback) in enumerate([("添加时段", lambda: self.edit_event(True)), ("修正", self.edit_event),
                                ("确认", lambda: self.review("已确认")), ("排除", lambda: self.review("已排除")),
                                ("复核本视频", self.review_video)]):
            actions.columnconfigure(column, weight=1)
            ttk.Button(actions, text=label, command=callback, width=0).grid(row=0, column=column, sticky="ew", padx=(0, 4))
        self.export_button = ttk.Button(actions, text="导出 CSV", command=self.export, style="Accent.TButton")
        self.export_button.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(5, 0))
        footer = ttk.Frame(self.root, padding=(18, 6, 18, 12))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(footer, maximum=100, length=180)
        self.progress.grid(row=0, column=1, sticky="e", padx=(12, 0))
        status_label = ttk.Label(footer, textvariable=self.status, wraplength=700, style="Hint.TLabel")
        status_label.grid(row=0, column=0, sticky="ew")
        footer.bind("<Configure>", lambda event: status_label.configure(wraplength=max(150, event.width - 240)))

    def selected_pack(self):
        index = self.model_combo.current()
        return self.packs[index] if 0 <= index < len(self.packs) else None

    def _refresh_models(self):
        self.model_combo["values"] = [f"{p.name} · {p.manifest.get('version', '1')}" for p in self.packs]
        if self.packs:
            self.model_combo.current(len(self.packs) - 1)
            self._model_changed()
        else:
            self.model_hint.configure(text="尚未导入模型，请点击右上方“导入模型包”。")

    def _model_changed(self, event=None):
        pack = self.selected_pack()
        if not pack:
            return
        p = pack.params
        self.full_params = p
        for key, variable in self.params.items():
            variable.set(p[key] / 60 if key == "min_duration" else p[key])
        descriptions = {"oviposition_v1": "检测 + ByteTrack + 净位移时序分析。每个视频选择一个最佳产卵候选。",
                        "detection_presence_v1": "按检测类别汇总出现时段。", "classification_v1": "逐帧分类后合并行为时段。"}
        self.model_hint.configure(text=descriptions[pack.manifest["pipeline"]])

    def import_model(self):
        path = filedialog.askopenfilename(title="导入识别模型包", filetypes=[("模型包", "*.behavior.zip *.zip")])
        if not path:
            return
        try:
            pack = install_pack(path, self.home / "models")
            if not any(p.identity == pack.identity for p in self.packs):
                self.packs.append(pack)
            self._refresh_models()
            self.model_combo.current(next(i for i, p in enumerate(self.packs) if p.identity == pack.identity))
            self._model_changed()
            self.status.set(f"已导入：{pack.name}。新分析会使用该模型包。")
        except Exception as exc:
            messagebox.showerror("导入失败", str(exc))

    def pack_dialog(self):
        window = tk.Toplevel(self.root)
        window.title("将 ONNX 封装成模型包")
        window.transient(self.root)
        window.grab_set()
        fields = {}
        defaults = [("模型文件", "weights", ""), ("模型名称", "name", "自定义识别模型"),
                    ("类别（英文逗号分隔，按模型顺序）", "labels", "locust"),
                    ("输入宽度", "width", "640"), ("输入高度", "height", "640"),
                    ("行为名称（产卵规则使用）", "behavior", "产卵"),
                    ("最短时段（秒）", "duration", "900")]
        for row, (label, key, default) in enumerate(defaults):
            ttk.Label(window, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=5)
            fields[key] = tk.StringVar(value=default)
            ttk.Entry(window, textvariable=fields[key], width=36).grid(row=row, column=1, padx=12, pady=5)
        ttk.Button(window, text="选择 ONNX", command=lambda: fields["weights"].set(filedialog.askopenfilename(filetypes=[("ONNX", "*.onnx")]))).grid(row=0, column=2, padx=8)
        pipeline = ttk.Combobox(window, state="readonly", values=["产卵时序分析", "检测类别出现时段", "逐帧分类时段"])
        pipeline.current(0)
        ttk.Label(window, text="识别方式").grid(row=7, column=0, sticky="w", padx=12)
        pipeline.grid(row=7, column=1, sticky="ew", padx=12)
        ttk.Label(window, text="检测支持 YOLO8/11 原始输出；分类默认 RGB、直接缩放、0–1 输入和概率输出。\n其他预处理请按模型包说明设置；不支持直接导入 PT / PKL。", wraplength=550, style="Hint.TLabel").grid(row=8, column=0, columnspan=3, padx=12, pady=10)
        def submit():
            try:
                choice = pipeline.current()
                labels = [v.strip() for v in fields["labels"].get().split(",")]
                out = filedialog.asksaveasfilename(title="保存模型包", defaultextension=".behavior.zip", filetypes=[("模型包", "*.behavior.zip")])
                if not out:
                    return
                create_pack(fields["weights"].get(), out, name=fields["name"].get(), labels=labels,
                            pipeline=["oviposition_v1", "detection_presence_v1", "classification_v1"][choice],
                            adapter="frame_classifier_onnx" if choice == 2 else "yolo_onnx",
                            input_size=(int(fields["width"].get()), int(fields["height"].get())),
                            behavior_label=fields["behavior"].get(), parameters={"min_duration": float(fields["duration"].get()), "stitch_gap": 0 if choice else 60})
                self.status.set(f"模型包已保存：{out}，可通过“导入模型包”使用。")
                window.destroy()
            except Exception as exc:
                messagebox.showerror("创建失败", str(exc), parent=window)
        ttk.Button(window, text="保存模型包", command=submit, style="Accent.TButton").grid(row=9, column=1, pady=12)

    def advanced(self):
        if not self.selected_pack():
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("高级分析参数")
        dialog.transient(self.root)
        fields = {}
        labels = {"win_sec": "净位移窗口 / 秒", "speed_percentile": "阈值兜底分位数", "max_drift": "最大漂移 / 像素",
                  "frag_min_duration": "最短碎片 / 秒", "stitch_gap": "拼接间隔 / 秒", "stitch_dist": "拼接距离 / 像素",
                  "min_track_len": "轨迹最少点数", "merge_gap": "静止段合并间隔 / 秒"}
        for row, (key, label) in enumerate(labels.items()):
            ttk.Label(dialog, text=label).grid(row=row, column=0, padx=12, pady=5, sticky="w")
            fields[key] = tk.StringVar(value=str(self.full_params[key]))
            ttk.Entry(dialog, textvariable=fields[key], width=16).grid(row=row, column=1, padx=12, pady=5)
        def apply():
            try:
                values = dict(self.full_params)
                values.update({key: float(var.get()) for key, var in fields.items()})
                self.full_params = validate_params(values)
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("参数无效", str(exc), parent=dialog)
        ttk.Button(dialog, text="应用", command=apply).grid(row=len(labels), column=1, pady=10)

    def add_videos(self):
        if self.running:
            return
        paths = filedialog.askopenfilenames(title="添加实验录像", filetypes=VIDEO_TYPES)
        for path in paths:
            resolved = str(Path(path).resolve())
            if resolved not in self.videos:
                self.videos.append(resolved)
        self._refresh_videos()
        if self.videos and self.current_video is None:
            self.video_list.selection_set(0)
            self._select_video()

    def remove_video(self):
        selected = self.video_list.curselection()
        if self.running or not selected:
            return
        path = self.videos.pop(selected[0])
        self.results.pop(path, None)
        self.dirty = True
        self._refresh_videos()
        if self.videos:
            self.video_list.selection_set(min(selected[0], len(self.videos) - 1))
            self._select_video()
        else:
            if self.cap:
                self.cap.release()
            self.cap, self.current_video, self.info = None, None, None
            self.preview.delete("all")
            self._refresh_events()

    def _refresh_videos(self):
        selected = self.video_list.curselection()
        self.video_list.delete(0, "end")
        for path in self.videos:
            result = self.results.get(path)
            label = f"{Path(path).name}  ·  {result['status'] if result else '待分析'}"
            self.video_list.insert("end", label)
        if selected and selected[0] < len(self.videos):
            self.video_list.selection_set(selected[0])

    def _select_video(self, event=None):
        selected = self.video_list.curselection()
        if not selected:
            return
        self.playing = False
        path = self.videos[selected[0]]
        if self.cap:
            self.cap.release()
        self.cap = None
        self.current_video = path
        try:
            self.info = video_info(path)
            self.cap = cv2.VideoCapture(path)
            self.slider.configure(to=max(0, self.info["duration"] - 1 / self.info["fps"]))
            self._load_preview_tracks()
            self.seek(0)
        except Exception as exc:
            self.info = None
            self.status.set(f"预览失败：{exc}")
        self._refresh_events()

    def _load_preview_tracks(self):
        self.preview_tracks = {}
        result = self.results.get(self.current_video, {})
        cache_path = result.get("cache_file", "")
        if cache_path and Path(cache_path).is_file():
            try:
                data = json.loads(Path(cache_path).read_text(encoding="utf-8"))
                self.preview_tracks = {key: ([p[0] for p in points], points) for key, points in data["tracks"].items()}
            except (ValueError, OSError, KeyError):
                pass

    def _slider_seek(self, value):
        if not getattr(self, "setting_slider", False):
            self.seek(float(value))

    def seek(self, position):
        if not self.cap or not self.info:
            return
        self.position = min(max(float(position), 0), max(0, self.info["duration"] - 1 / self.info["fps"]))
        self.setting_slider = True
        self.slider.set(self.position)
        self.setting_slider = False
        self.show_frame()
        self.draw_timeline()

    def show_frame(self):
        if not self.cap or not self.info:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, round(self.position * self.info["fps"]))
        ok, frame = self.cap.read()
        if not ok:
            return
        if self.overlay.get():
            result = self.results.get(self.current_video, {})
            tolerance = .6 / result.get("parameters", {}).get("sample_fps", 1)
            for tid, (times, points) in self.preview_tracks.items():
                index = bisect.bisect_left(times, self.position)
                choices = [i for i in [index - 1, index] if 0 <= i < len(times)]
                if not choices:
                    continue
                index = min(choices, key=lambda i: abs(times[i] - self.position))
                if abs(times[index] - self.position) > tolerance:
                    continue
                _, cx, cy, w, h = points[index]
                cv2.rectangle(frame, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (130, 230, 65), 2)
                cv2.putText(frame, f"ID {tid}", (int(cx - w / 2), max(12, int(cy - h / 2) - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (130, 230, 65), 1)
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((max(10, self.preview.winfo_width() - 12), max(10, self.preview.winfo_height() - 12)))
        self.photo = ImageTk.PhotoImage(image)
        self.preview.delete("all")
        self.preview.create_image(self.preview.winfo_width() / 2, self.preview.winfo_height() / 2, image=self.photo)
        self.time_text.set(f"{format_time(self.position)} / {format_time(self.info['duration'])}")

    def toggle_play(self):
        if not self.cap:
            return
        self.playing = not self.playing
        if self.playing:
            self.play_time, self.play_position = time.monotonic(), self.position
            self._play_tick()

    def _play_tick(self):
        if not self.playing or not self.info:
            return
        target = self.play_position + time.monotonic() - self.play_time
        self.seek(target)
        if target >= self.info["duration"] - 1 / self.info["fps"]:
            self.playing = False
        else:
            self.root.after(100, self._play_tick)

    def draw_timeline(self):
        self.timeline.delete("all")
        if not self.info:
            return
        width, duration = max(20, self.timeline.winfo_width() - 16), self.info["duration"]
        self.timeline.create_rectangle(8, 10, width + 8, 32, fill="#e9edf5", outline="")
        for event in self.results.get(self.current_video, {}).get("events", []):
            color = "#abb5c5" if event["review"] == "已排除" else "#4f70e6"
            self.timeline.create_rectangle(8 + event["start"] / duration * width, 10, 8 + event["end"] / duration * width, 32, fill=color, outline="")
        x = 8 + self.position / duration * width
        self.timeline.create_line(x, 6, x, 38, fill="#ed9346", width=2)
        self.timeline.create_text(8, 46, text="0:00", anchor="w", fill="#66758c")
        self.timeline.create_text(width + 8, 46, text=format_time(duration), anchor="e", fill="#66758c")

    def _timeline_seek(self, event):
        if self.info:
            self.seek((event.x - 8) / max(20, self.timeline.winfo_width() - 16) * self.info["duration"])

    def _refresh_events(self):
        self.event_table.delete(*self.event_table.get_children())
        result = self.results.get(self.current_video, {})
        for index, event in enumerate(result.get("events", [])):
            self.event_table.insert("", "end", iid=str(index), values=(event["behavior"], format_time(event["start"]),
                                    format_time(event["end"]), f"{(event['end'] - event['start']) / 60:.2f}", event["review"]))
        self.draw_timeline()

    def _selected_event(self):
        selected = self.event_table.selection()
        return self.results[self.current_video]["events"][int(selected[0])] if selected else None

    def _jump_event(self, ignored=None):
        event = self._selected_event()
        if event:
            self.playing = False
            self.seek(event["start"])

    def edit_event(self, new=False):
        if self.running or not self.info:
            return
        event = None if new else self._selected_event()
        if not new and not event:
            return
        dialog = tk.Toplevel(self.root)
        dialog.title("添加行为时段" if new else "修正行为时段")
        dialog.transient(self.root)
        dialog.grab_set()
        pack = self.selected_pack()
        default_label = pack.manifest.get("behavior_label", "行为") if pack else "行为"
        values = [event["behavior"] if event else default_label, format_time(event["start"] if event else self.position),
                  format_time(event["end"] if event else min(self.position + 60, self.info["duration"])), event.get("note", "") if event else ""]
        fields = []
        for row, (label, value) in enumerate(zip(["行为", "起点（时:分:秒）", "终点（时:分:秒）", "复核备注"], values)):
            ttk.Label(dialog, text=label).grid(row=row, column=0, padx=12, pady=6, sticky="w")
            var = tk.StringVar(value=value)
            ttk.Entry(dialog, textvariable=var, width=32).grid(row=row, column=1, padx=12, pady=6)
            fields.append(var)
        def apply():
            try:
                behavior, start, end, note = [v.get().strip() for v in fields]
                start, end = parse_time(start), parse_time(end)
                validate_interval(start, end, self.info["duration"])
                if not behavior:
                    raise ValueError("行为名称不能为空")
                result = self.results.setdefault(self.current_video, dict(video=self.current_video, status="人工标注", events=[],
                                                    info=self.info, model_name="", model_id="", parameters={}, error=""))
                result.setdefault("info", self.info)
                if new:
                    result["events"].append(dict(start=start, end=end, behavior=behavior, note=note, review="已确认", source="人工添加", original={}, support=0, track_ids=[]))
                    result["status"] = "人工补充" if result.get("model_id") else "人工标注"
                else:
                    event.update(start=start, end=end, behavior=behavior, note=note, review="已修正")
                self.dirty = True
                self._refresh_events()
                self._refresh_videos()
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("无法保存", str(exc), parent=dialog)
        ttk.Button(dialog, text="保存时段", command=apply, style="Accent.TButton").grid(row=4, column=1, pady=12)

    def review(self, status):
        if self.running:
            return
        event = self._selected_event()
        if event:
            event["review"] = status
            self.dirty = True
            self._refresh_events()

    def review_video(self):
        if self.running or self.current_video not in self.results:
            return
        result = self.results[self.current_video]
        if result["status"] in {"出错", "已停止"}:
            self.status.set("当前视频未完成分析，可添加人工时段或重新分析。")
            return
        result["video_review"] = "已复核"
        for event in result["events"]:
            if event["review"] == "待复核":
                event["review"] = "已确认"
        self.dirty = True
        self._refresh_events()
        self.status.set("当前视频已标记复核完成。")

    def start_analysis(self):
        pack = self.selected_pack()
        if self.running:
            return
        if not pack or not self.videos:
            messagebox.showinfo("准备分析", "请先导入模型包并添加视频。")
            return
        try:
            params = dict(self.full_params)
            params.update({key: var.get() * 60 if key == "min_duration" else var.get() for key, var in self.params.items()})
            params = validate_params(params)
        except Exception as exc:
            messagebox.showerror("参数无效", str(exc))
            return
        self.running = True
        self.cancel.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.model_combo.configure(state="disabled")
        videos = list(self.videos)
        cache_dir = self.home / "cache" if self.cache.get() else None
        def worker():
            try:
                for index, path in enumerate(videos):
                    if self.cancel.is_set():
                        break
                    def progress(fraction, text):
                        self.messages.put(("progress", ((index + fraction) / len(videos), f"{index + 1}/{len(videos)} {Path(path).name} · {text}")))
                    try:
                        result = analyze(pack, path, params, cache_dir, self.cancel, progress)
                    except InterruptedError:
                        self.messages.put(("status", "已停止；已完成的视频结果保留。"))
                        break
                    except Exception as exc:
                        result = dict(video=path, status="出错", events=[], error=str(exc), model_name=pack.name,
                                      model_id=pack.identity, model_path=str(pack.root), parameters=params)
                    self.messages.put(("result", result))
            finally:
                self.messages.put(("done", self.cancel.is_set()))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, payload = self.messages.get_nowait()
                if kind == "progress":
                    self.progress["value"] = payload[0] * 100
                    self.status.set(payload[1])
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "result":
                    previous = self.results.get(payload["video"])
                    if previous:
                        previous = dict(previous)
                        history = previous.pop("previous_runs", [])
                        payload["previous_runs"] = history + [previous]
                    self.results[payload["video"]] = payload
                    self.dirty = True
                    self._refresh_videos()
                    if self.current_video == payload["video"]:
                        self._load_preview_tracks()
                        self._refresh_events()
                        self.show_frame()
                elif kind == "done":
                    self.running = False
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.model_combo.configure(state="readonly")
                    if not payload:
                        self.progress["value"] = 100
                        errors = sum(r["status"] == "出错" for r in self.results.values())
                        self.status.set(f"批量分析结束 · {errors} 个视频出错。请复核识别时段，也检查未检出的视频。")
                    else:
                        self.status.set("已停止；已完成的视频结果保留。")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def save(self):
        if self.running:
            self.status.set("请在分析结束或停止后保存项目。")
            return False
        path = filedialog.asksaveasfilename(title="保存分析项目", defaultextension=".behavior.json", filetypes=[("分析项目", "*.behavior.json")])
        if path:
            try:
                rows = [self.results.get(v, dict(video=v, events=[], status="待分析", error="")) for v in self.videos]
                pack = self.selected_pack()
                save_project(path, rows, pack.root if pack else "")
                self.dirty = False
                self.status.set(f"项目已保存：{path}")
                return True
            except Exception as exc:
                messagebox.showerror("保存失败", str(exc))
        return False

    def _allow_replace(self):
        if self.running:
            return False
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("保存项目", "分析结果或复核修改尚未保存，是否先保存项目？")
        return self.save() if answer else answer is False

    def open_project(self):
        if not self._allow_replace():
            return
        path = filedialog.askopenfilename(title="打开分析项目", filetypes=[("分析项目", "*.behavior.json *.json")])
        if not path:
            return
        try:
            data = load_project(path)
            self.results = {r["video"]: r for r in data["results"]}
            self.videos = list(self.results)
            self.dirty = False
            self._refresh_videos()
            if data.get("model_path"):
                try:
                    pack = load_pack(data["model_path"])
                    if not any(p.identity == pack.identity for p in self.packs):
                        self.packs.append(pack)
                    self._refresh_models()
                    self.model_combo.current(next(i for i, p in enumerate(self.packs) if p.identity == pack.identity))
                    self._model_changed()
                except (OSError, ValueError):
                    pass
            if self.videos:
                self.video_list.selection_set(0)
                self._select_video()
            self.status.set("项目已打开。若移动了视频，请将文件恢复到项目记录的位置。")
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def export(self):
        if self.running or not self.results:
            return
        path = filedialog.asksaveasfilename(title="导出结果", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                export_csv(path, list(self.results.values()))
                self.status.set("CSV 已导出。包含未检出/出错记录、复核状态及原始预测；已排除时段也保留供追溯。")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))

    def close(self):
        if self.running:
            self.cancel.set()
            self.status.set("正在停止分析，请稍后关闭窗口。")
            return
        if self._allow_replace():
            self.playing = False
            if self.cap:
                self.cap.release()
            self.root.destroy()
