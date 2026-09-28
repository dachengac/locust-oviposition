# locust-oviposition

飞蝗产卵行为自动分析。输入实验录像，输出每段产卵的起止时刻与时长。

基于 YOLO11 检测 + ByteTrack 追踪 + 时序分析。在国产海光 DCU 上开发，同样适用于 NVIDIA GPU。

## 精度

12 个视频对照人工标注：

| 指标 | 平均绝对误差 | 中位数 |
|------|-------------|--------|
| 产卵起点 | 2.3 min | 1.1 min |
| 产卵终点 | 5.0 min | 3.6 min |
| 产卵时长 | 6.1 min | 3.2 min |

检出率 12/12，其中 10 个视频起点误差 ≤ 1.4 min。检测器 mAP50 = 0.983。

## 原理

产卵时腹部插入基质被遮挡，单帧不可见，因此不依赖姿态识别，而是：

1. YOLO 检测飞蝗 → ByteTrack 追踪
2. 计算**净位移率**（窗口首尾直线距离 / 时长）——产卵时原地晃动，净位移≈0；爬行时净位移大
3. Otsu 自适应阈值二值化，不写死像素值
4. 轨迹拼接修复 ID 断裂，按"支持轨迹数 × 时长"选择候选时段

> 不能用逐帧速度：实测产卵段逐帧速度（5.07 px/s）反而高于爬行段（3.62 px/s），因为晃动每帧改变方向。

## 安装

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
    --win-sec 180 --frag-min-duration 900 \
    --stitch-gap 60 --stitch-dist 100

# 有人工标注时可对比误差
python scripts/detect_oviposition.py ... --stages stages.csv
```

轨迹缓存后调参无需重新推理。

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

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--conf` | 0.7 | 检测置信度。低于 0.5 会在反光区产生大量假框 |
| `--sample-fps` | 1.0 | 抽帧率，产卵为慢行为，1 fps 足够 |
| `--win-sec` | 180 | 净位移计算窗口 |
| `--min-duration` | 900 | 最短产卵时长（秒） |
| `--stitch-gap` | 60 | 轨迹拼接最大时间间隔（秒） |
| `--stitch-dist` | 100 | 轨迹拼接最大空间距离（像素） |

参数在 12 个视频上搜索，前 20 组结果一致，对参数不敏感。

## License

MIT
