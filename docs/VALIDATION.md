# 本次封装验证记录

0.1.1 布局修复增加 9 组窗口/显示缩放回归：900×560、1000×600、1180×640，
分别测试 100%、125%、150% 缩放。检查开始/停止/导出/进度可见，以及左侧设置可滚动访问。
桌面流程测试通过实际“开始批量分析”按钮启动后台任务。

## 通过的检查

* 13 项自动测试：模型包导入/重复导入、路径与校验、参数范围、追踪匹配、短暂丢失与低分检测恢复、缓存失效、取消任务、实际 ONNX 接口、时间格式、项目和 CSV 原始预测记录。
* 使用同一组模拟轨迹，封装后的产卵时间段与原 `analyze_video` 起点/终点一致。
* 已发布 `best.pt` SHA-256 与 GitHub 附件摘要一致。
* PyTorch 与 OpenCV ONNX 的三幅确定性测试画面：最大检测框差异 0.0014953613 像素，最大类别分数差异 0.0000038445。
* Tk 桌面交互流程：预览、后台批处理、时间轴、事件表、确认、项目保存/重新打开、CSV 导出。
* Windows EXE：默认模型加载、CPU 推理、窗口布局、无训练依赖加载、视频解码/批量分析与 CSV 输出。
* 完整 ZIP 校验，并解压到带中文的新目录后测试独立 EXE。

详细 JSON 位于源码 `assets/verification/`，发布包包含构建时已有的报告。

## 验证边界

上述验证使用合成画面、模拟轨迹和软件流程测试，未使用带人工标注的真实实验录像重测行为识别准确率。
原项目的 12 视频精度指标属于原后端，不作为此次桌面版的实测指标。
模拟分类器仅用于测试软件接口，没有作为识别模型放入发布包。
桌面版使用 NumPy ByteTrack 与 OpenCV ONNX，可能存在追踪匹配与采样差异。
正式实验前请用有人工标注的录像比较时段起止，尤其检查未检出的视频。

## 可重复命令

```powershell
python -m unittest discover -s tests -v
python tools/verify_desktop.py
python tools/verify_export.py --weights assets/original/best.pt --pack path/to/unpacked-model --report parity.json
BehaviorLab.exe --self-test smoke.json
BehaviorLab.exe --analyze video.avi --output results
```

转换比较工具在独立开发环境运行；免安装软件只需要解压后的运行库。
