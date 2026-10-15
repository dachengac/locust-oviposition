# locust-oviposition · BehaviorLab

飞蝗产卵行为分析项目，包含原有训练/服务器脚本和 **BehaviorLab 0.1.1 Windows 桌面版**。
默认流程是飞蝗检测 → ByteTrack 追踪 → 净位移时序分析。
每段录像自动选择一个最佳产卵候选，输出起点、终点和时长；用户可在界面中复核、修正或添加时段。

**桌面版已做软件功能与模型转换测试，尚未在有人工标注的实验录像上重新评估识别准确率。**
原项目的历史精度不能直接视为桌面版的实测精度。
原训练/服务器流程使用 Ultralytics/PyTorch；Windows 桌面版使用 CPU ONNX 推理。

## Windows 桌面软件（BehaviorLab）

进入 [Releases](https://github.com/dachengac/locust-oviposition/releases)，找到 **BehaviorLab 0.1.1**，在 Assets 中选择：

| 文件 | 用途 | 下载大小（约） |
|---|---|---:|
| `BehaviorLab-Windows-x64-0.1.1.zip` | 首次使用推荐；包含程序、运行库和默认飞蝗模型 | 103 MB |
| `BehaviorLab-0.1.1-Update.zip` | 更新此前私下分发的 0.1.0 初版；不能单独运行 | 12 MB |
| `locust-oviposition.behavior.zip` | 独立飞蝗模型包，供软件导入 | 33 MB |
| `BehaviorLab-Software-0.1.1.zip` | 软件本体；需要另外导入兼容模型包 | 71 MB |
| `SHA256SUMS.txt` | 发布文件完整性校验 | — |

GitHub 自动提供的 **Source code** 是源码，不包含可运行的 EXE 或默认模型。
发布文件的精确大小及 SHA-256 见对应发布附件和校验文件。1 MB = 1,000,000 字节。
完整版全部解压后约占 230 MB；录像、其他模型、项目和缓存另外占空间。
这是 Windows x64 免安装版，在 Windows 11 x64 上测试过；其他 Windows 版本未单独验证。
程序使用 CPU 运行，无需安装 Python、PyTorch 或 CUDA。

1. 将完整版 ZIP **全部解压**到可写文件夹。
2. 打开 `BehaviorLab.exe`，保留同级 `_internal` 和 `models`。
3. 点击“添加视频”，选择实验录像，可多选。
4. 点击**窗口顶部蓝色“开始批量分析”**，也可按 F5。
5. 选择录像、双击事件查看对应时段；用“修正”“确认”“排除”或“添加时段”复核。
6. “保存项目”便于以后继续复核，“导出 CSV”保存结果表格。

更新旧版时，先保存项目并关闭软件，再将更新包解压到旧版目录、覆盖同名文件。
更新包不包含 `data/`、模型或原视频。首次使用请下载完整版。

软件支持兼容接口的 YOLO8/11 检测模型，以及 RGB 单帧互斥分类模型；不包括模型训练。
更换动物检测器不代表自动获得新的动作识别能力，相应规则也需正确配置。
姿态、分割、视频片段网络、随机森林 PKL 等需要新的适配器。

源码使用带 Tk 的 Python（本次构建使用 Python 3.12）：

```powershell
python -m pip install -r requirements-desktop.txt
python desktop_app.py
```

从 Releases 下载独立模型包后，可在界面导入，或放入源码的 `models/` 目录。
源码仓库不存放训练权重、完整运行库、原始录像或用户缓存。
说明见 [桌面版操作](docs/DESKTOP.md)、[模型包接口](docs/MODEL_PACK.md)、[验证记录](docs/VALIDATION.md)。

## 耗时、内存与过程文件

默认约每秒抽取 1 帧做模型推理，逐帧读取视频，不导出整套图片，也不复制原视频。
批量任务逐个运行；轨迹和抽样记录仍随录像长度、目标数量增长。
开启缓存时，每个视频/模型/识别设置组合生成一个 JSON 轨迹缓存，保存在软件旁的 `data/cache/`。
缓存目前没有自动容量上限。关闭软件或停止分析后可清理该目录；取消缓存选项不会删除已有缓存。

本机 30 次模拟画面推理测试平均约 0.178 秒/帧：1 小时录像按 1 fps 抽样，**仅模型计算**约 10.7 分钟，
不包括整段录像的校验、解码、追踪、时序计算和界面开销。
该测试物理内存峰值约 266 MB、系统内存提交峰值约 1.02 GB；这不是完整真实录像的内存上限。
机器、录像编码和目标数量都会影响耗时与内存，不能作为所有电脑的性能保证。
原始记录见 [performance-estimate.json](assets/verification/performance-estimate.json)。

## 原项目报告的历史精度

来源：[原项目 v1.0](https://github.com/dachengac/locust-oviposition/tree/v1.0)。
原项目报告 12 个视频对照人工标注的结果如下，本次桌面封装没有独立复现这些实验：

| 指标 | 平均绝对误差 | 中位数 |
|------|-------------|--------|
| 产卵起点 | 2.3 min | 1.1 min |
| 产卵终点 | 5.0 min | 3.6 min |
| 产卵时长 | 6.1 min | 3.2 min |

原项目另报告检出率 12/12，其中 10 个视频起点误差 ≤ 1.4 min，检测器 mAP50 = 0.983。
这些指标的样本、任务和运行后端与本次桌面软件功能测试不同。

## 原理

产卵时腹部插入基质被遮挡，单帧不可见，因此不依赖姿态识别，而是：

1. YOLO 检测飞蝗 → ByteTrack 追踪
2. 计算**净位移率**（窗口首尾直线距离 / 时长）——产卵时原地晃动，净位移≈0；爬行时净位移大
3. Otsu 自适应阈值二值化，不写死像素值
4. 轨迹拼接修复 ID 断裂，先比较支持轨迹数，再比较时长，选择一个最佳候选时段

> 不能用逐帧速度：实测产卵段逐帧速度（5.07 px/s）反而高于爬行段（3.62 px/s），因为晃动每帧改变方向。

## 原有训练/服务器环境安装

```bash
conda create -n locust python=3.10 -y
conda activate locust
pip install -r requirements.txt
```

海光 DCU 环境需用平台预编译的 dtk 版 PyTorch，见 [docs/ENV_SETUP.md](docs/ENV_SETUP.md)。

## 使用

### 图形界面

```bash
python app.py --weights path/to/best.pt
```

浏览器访问 `http://127.0.0.1:7860`。远程服务器用 SSH 端口转发：

```bash
ssh -L 7860:127.0.0.1:7860 user@host
```

### 命令行

```bash
# 批量分析
python scripts/detect_oviposition.py \
    --weights best.pt --video-dir videos/ \
    --cache-tracks track_cache --out results.csv \
    --win-sec 180 --speed-percentile 25 --frag-min-duration 900 \
    --stitch-gap 60 --stitch-dist 100

# 有人工标注时可对比误差
python scripts/detect_oviposition.py ... --stages stages.csv
```

轨迹缓存后调时序参数可跳过重新推理。原服务器脚本的缓存规则与桌面版不同；
更换权重时应使用新的服务器轨迹缓存目录。

## 训练自己的检测器

```bash
# 1. 按标注时段抽帧
python scripts/extract_3stage.py --videos videos/ --stages stages.csv \
    --out frames --per-stage 40

# 2. 标注（Roboflow 等），导出 YOLO 格式后整理数据集
python scripts/prep_singleclass.py --in labeled/train --out dataset \
    --val-videos v03,v07,v10

# 3. 训练
sbatch scripts/slurm/train.sh
```

`--val-videos` 按视频划分训练/验证集，避免相邻帧同时进入两端导致精度虚高。

## 脚本

| 脚本 | 用途 |
|------|------|
| `app.py` | 图形界面 |
| `detect_oviposition.py` | 核心：检测 + 追踪 + 时序分析 |
| `extract_3stage.py` | 按标注时段抽帧 |
| `prep_singleclass.py` | 数据集整理，按视频划分 |
| `sweep_params.py` | 参数网格搜索（读缓存，秒级） |
| `diagnose.py` | 单视频诊断，ASCII 净位移曲线 |
| `export_clips.py` | 导出带框验证片段 |
| `analyze_groups.py` | 两组行为指标统计比较 |
| `behavior_classifier.py` | 通用行为分类（实验性，见下） |

## 标注格式

```csv
video,explore_start,explore_end,ovi_start,ovi_end,clean_start,clean_end
v01,0:00:00,0:18:51,0:18:51,1:32:53,1:32:53,1:42:19
```

## 适用范围

**适用**：静止型长时程行为。

**不适用**：精细动作区分。`behavior_classifier.py` 尝试用同一套轨迹特征区分探索/产卵/清洁，macro F1 仅 0.5（三分类随机 0.33），探索有 62% 被误判为产卵——检测框不含"腹部是否插入基质"的信息。区分精细动作需关键点姿态估计。

**已知局限**：
- 探索期长时间原地停留会使起点判早（最差 -17.5 min）
- 产卵末期追踪丢失会使终点判早（最差 -17.2 min）
- 未检出产卵的视频需人工复核

## 默认桌面模型包参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--conf` | 0.7 | 检测置信度。低于 0.5 会在反光区产生大量假框 |
| `--sample-fps` | 1.0 | 抽帧率，产卵为慢行为，1 fps 足够 |
| `--win-sec` | 180 | 净位移计算窗口 |
| `--min-duration` | 900 | 最短产卵时长（秒） |
| `--stitch-gap` | 60 | 轨迹拼接最大时间间隔（秒） |
| `--stitch-dist` | 100 | 轨迹拼接最大空间距离（像素） |

这些默认值沿用原项目定版配置。不同 CLI 的实际默认参数以其 `--help` 为准。
原项目报告的参数搜索结论尚未在桌面推理后端重新验证。

## License

应用源码使用 [MIT](LICENSE)。第三方运行库与模型来源信息见 [THIRD_PARTY.md](THIRD_PARTY.md)，
各组件仍适用其随附许可证，不能将所有第三方内容一概视为 MIT。
