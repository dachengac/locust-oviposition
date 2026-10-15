# 可替换模型包接口 v1

一个包是 ZIP 文件，建议后缀 `.behavior.zip`。根目录包含 `manifest.json` 与 `model.onnx`。
不包含 Python 插件或需要执行的脚本。导入会校验版本、路径、类别与 SHA-256。

```json
{
  "schema_version": 1,
  "name": "飞蝗产卵",
  "version": "1.0.0",
  "adapter": "yolo_onnx",
  "pipeline": "oviposition_v1",
  "weights": "model.onnx",
  "sha256": "替换成模型文件的64位小写SHA256",
  "labels": ["locust"],
  "input_size": [640, 640],
  "output_layout": "channels_first",
  "target_class": 0,
  "behavior_label": "产卵",
  "parameters": {"conf": 0.7, "sample_fps": 1, "min_duration": 900}
}
```

`labels` 是检测类别列表，顺序必须对应模型通道。产卵规则分析 `target_class` 的轨迹，输出 `behavior_label`。
输入宽高是固定模型尺寸，不是原始录像尺寸。

## 支持的适配器与分析规则

| adapter | pipeline | 输入/输出约定 | 含义 |
|---|---|---|---|
| `yolo_onnx` | `oviposition_v1` | float32 RGB NCHW `[1,3,H,W]`；输出 `[1,4+C,N]` | 检测、追踪、静止行为规则；每视频最佳候选 |
| `yolo_onnx` | `detection_presence_v1` | 同上 | 按类别合并画面中出现时段，非个体行为分类 |
| `frame_classifier_onnx` | `classification_v1` | float32 RGB NCHW `[1,3,H,W]`；输出 `[1,C]` | 逐帧最高概率类别，再按时间合并 |

检测预处理：保持比例 letterbox（填充值114），RGB / 255，框为输入图像像素 `cx,cy,w,h`。
后处理：类别概率阈值、分类别 NMS（IoU 0.7）、最多 300 框、反变换到原视频坐标。
`output_layout="rows"` 可使用 `[1,N,4+C]`。不支持 YOLO5 objectness 通道、
YOLO26 end-to-end、NMS 内嵌输出、分割/姿态输出、动态批量和 ONNX 外部权重文件。
YOLO 导出建议 `opset=12, nms=False, dynamic=False, half=False, simplify=False, batch=1`。

分类预处理：直接缩放到固定宽高，RGB / 255，之后 `(pixel - mean) / std`。
`mean`、`std` 分别默认 `[0,0,0]`、`[1,1,1]`。
`output_kind="probabilities"` 直接使用概率；`"logits"` 先做 softmax。
目前只支持互斥类别的最高概率类别，支持低置信度时不给类别。
需要 center crop、多标签 sigmoid、关键点或视频片段输入时，应新增适配器与相应规则。

## 命令行创建模型包

```powershell
# 新的动物检测器，沿用产卵/静止时序规则
python tools/model_pack.py --weights detector.onnx --output animal.behavior.zip --name "动物静止行为" --labels animal --behavior "静止"

# 已训练的 RGB 单帧行为分类器；类别顺序与模型相同
python tools/model_pack.py --weights classifier.onnx --output classifier.behavior.zip --name "行为分类" --labels explore rest clean --pipeline classification_v1 --input-size 224 224 --output-kind logits --mean 0.485 0.456 0.406 --std 0.229 0.224 0.225 --parameters '{"conf":0.7,"min_duration":5,"stitch_gap":0}'
```

更换参数可填写 `conf`、`sample_fps`、`win_sec`、`speed_percentile`、`max_drift`、
`min_duration`、`frag_min_duration`、`stitch_gap`、`stitch_dist`、`min_track_len`、`merge_gap`。
时间均以秒为单位；GUI 中最短时段以分钟显示。默认值来自原项目定版参数。

## 缓存与结果接口

缓存依据：完整视频 SHA-256、模型 SHA-256 和 manifest、推理版本、抽帧率、置信度。
修改置信度、抽帧率、视频内容或模型会重跑推理；仅在 GUI 调整时序参数时复用检测轨迹。
缓存以 JSON 保存，损坏会重新计算；读取不执行 pickle。

分析返回录像信息、模型标识、完整参数、事件列表与状态。
事件提供 `start/end`（秒）、`behavior`、`track_ids`、`support`、`review`、`source`、`original`、`note`。
人工修正保留 `original`；重新分析在项目的 `previous_runs` 中保留此前结果。
图形软件不负责训练，训练好的模型按此接口导入。
