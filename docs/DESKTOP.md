# BehaviorLab · Windows 免安装版

0.1.1：将蓝色“开始批量分析”和“停止分析”固定在窗口顶部，支持 F5 开始；
左侧设置可滚动，窗口适配较小屏幕与显示缩放。

软件界面与模型包分开。软件负责视频、任务队列、预览、人工复核和导出；
模型包负责识别权重、类别、输入输出接口、分析规则与默认参数。

## 使用

1. 从 [GitHub Releases](https://github.com/dachengac/locust-oviposition/releases) 下载完整版 `BehaviorLab-Windows-x64-0.1.1.zip`，全部解压到可写目录。GitHub 的 `Source code` 是源码，不能直接运行。
2. 双击整个目录里的 `BehaviorLab.exe`。保留旁边的 `_internal`、`models` 和说明文件。
3. 默认载入来自 GitHub v1.0 发布附件的飞蝗模型。
4. 点击“添加视频”，可多选 MP4、AVI、MOV、MKV 录像。
5. 点击窗口顶部蓝色“开始批量分析”（或按 F5）。CPU 本地离线运行，首次分析较慢。选择“停止分析”会保留已完成的视频。
6. 选择视频，播放/暂停，点击时间轴，或双击行为时段跳到起点；缓存开启时可查看抽样轨迹框。
7. 使用“修正”“确认”“排除”或“添加时段”校对。“复核本视频”也可给未检出的视频标记已复核。
8. “保存项目”保存录像路径、参数、预测与复核记录。“打开项目”可继续工作。
9. “导出 CSV”生成 Excel 可读的 UTF-8 文件，包括原始预测、人工修正、模型标识和分析错误。

CSV 中“已排除”的事件仍保留供追溯，做统计时请按复核状态过滤。
未检出与处理出错分别记录，不等于“没有该行为”。
项目引用原视频路径，不复制大视频；移动原视频后应恢复原位置。
缓存、已导入模型在软件旁边的 `data/` 下；项目与 CSV 保存位置由你选择。
清理缓存可删除 `data/cache/`，下次会重新推理。
播放预览为静音视频，不提供音轨分析。

更新包 `BehaviorLab-0.1.1-Update.zip` 只用于此前私下分发的 0.1.0 初版。
先保存项目并关闭旧软件，再解压到原软件目录覆盖同名文件，重新打开保存的项目。
第一次使用需下载完整版，更新包不能单独运行。

本次软件在 Windows 11 x64 上测试过，其他 Windows 版本未单独验证。
缓存目前没有自动容量限制。取消“复用检测与轨迹缓存”不会删除已有文件；
录像更长、目标更多或反复更换模型/识别参数时，轨迹文件和运行内存也会增加。
CSV 中的原始预测属于模型输出，不会自动成为人工标注的真值。

## 更换识别模型

点击“导入模型包”，选择 `*.behavior.zip`，在“识别模型”中选择新包。
新的任务会采用新模型；已有结果保留自己的模型标识。
支持同接口 YOLO8/11 检测模型和符合约定的逐帧分类模型。

如果已有兼容的 ONNX，可在“创建模型包”中填写类别与输入尺寸、选择分析方式后保存。
类别顺序必须与模型输出一致。复杂归一化配置可通过开发工具指定，见 [MODEL_PACK.md](MODEL_PACK.md)。

如果现在只有 `best.pt`，在已有训练环境中运行：

```powershell
python tools/export_model.py --weights path/to/best.pt --output my-model.behavior.zip
```

转换工具仅在训练/开发环境使用。软件不包含 PyTorch 或训练功能，以减少体积。
不要将其他类型模型仅重命名成 ONNX 或 ZIP；需遵守输入输出接口。
姿态模型、视频片段时序网络、随机森林 PKL 需要新适配器后才能接入；
当前程序不把低精度实验性三分类 PKL 默认包装成可靠识别模型。

## 当前飞蝗包的行为解释

复用原项目的净位移率、Otsu 阈值、静止段筛选与轨迹拼接算法，
按支持轨迹数优先、时长其次，每段录像返回一个最佳产卵候选。
飞蝗检测类别与“产卵行为”分开：模型看见飞蝗，再由长时程轨迹推断产卵。
换成其他动物检测器并不自动获得新的动作识别能力，行为规则也应放在模型包内正确配置。

轻量版使用 OpenCV ONNX CPU 推理与 NumPy 版本 ByteTrack。
ByteTrack 的两阶段匹配、Kalman 状态保留；去掉 PyTorch、SciPy 和 LAP 运行依赖。
相对于原 Ultralytics 后端，分配实现、浮点细节和采样时刻可能不同。
轻量版以实际抽到的视频帧索引计算时刻，第一帧为 0 秒。
原仓库报告的 12 视频准确率未在此部署后端重新验证，不能作为新版实测结果。
软件可做复核辅助，正式数据使用前需校对事件和未检出录像。

## 体积策略

使用 Windows 原生 Tk 界面、NumPy、OpenCV DNN、Pillow。
没有浏览器运行时、Gradio、训练框架、CUDA 或 GPU 驱动包。
运行库与模型分开，今后只发模型包即可更新识别。
发布采用文件夹免安装形式，避免单文件 EXE 每次启动解压整个运行库。
文件夹内 `SIZE_REPORT.json` 记录解压体积；发布附件的 `SIZE_REPORT-0.1.1.json` 和 `SHA256SUMS.txt`
记录 ZIP 大小与完整性校验。大小使用十进制字节，1 MB = 1,000,000 字节。

## 从源码启动或重新打包

```powershell
python -m pip install -r requirements-desktop.txt
python desktop_app.py
# 开发环境打包
python -m pip install -r requirements-build.txt
powershell -ExecutionPolicy Bypass -File tools/build_windows.ps1 -Python path/to/lean/python.exe
```

请在独立轻量环境构建，不要在含 PyTorch 的训练环境直接打包。
模型转换环境需要 `ultralytics`、CPU `torch`、`onnx`；它们不进入最终软件。
命令行也支持 `desktop_app.py --model package.behavior.zip --analyze video.mp4 --output results`。

## 参考与许可

流程参考 [BORIS 视频编码与事件表](https://www.boris.unito.it/user_guide/coding/) 和
[SimBA 项目、模型运行与结果可视化](https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md)。
界面为本项目独立实现。追踪组件基于 [ByteTrack](https://github.com/ifzhang/ByteTrack)，MIT 许可证随软件提供。
源模型取自 [locust-oviposition v1.0](https://github.com/dachengac/locust-oviposition/releases/tag/v1.0)。
项目代码与依赖许可证分别列于 `LICENSE`、`THIRD_PARTY.md`；运行版的 `licenses/` 保留依赖许可证。
