# 海光 DCU 环境配置

国产海光 DCU 使用 ROCm/HIP 生态（非 NVIDIA CUDA），需用平台预编译的 dtk 版 PyTorch。
NVIDIA GPU 用户按常规 `pip install ultralytics` 即可，跳过本文。

实测环境：曙光智算 wzhdtest 分区，DTK 24.04，torch 2.1.0，Python 3.10。

## 每次使用前的环境激活

```bash
module purge
module load compiler/devtoolset/7.3.1      # gcc 运行时
module load mpi/hpcx/2.11.0/gcc-7.3.1      # libmpi.so.40，torch 依赖
module load compiler/dtk/24.04             # DCU 工具链，libgalaxyhip.so.5
export LD_LIBRARY_PATH=$HOME/openssl11/lib:$LD_LIBRARY_PATH   # libssl.so.1.1
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate locust
```

建议写成 alias：

```bash
echo "alias locust-env='module purge && module load compiler/devtoolset/7.3.1 && module load mpi/hpcx/2.11.0/gcc-7.3.1 && module load compiler/dtk/24.04 && export LD_LIBRARY_PATH=\$HOME/openssl11/lib:\$LD_LIBRARY_PATH && source \$HOME/miniconda3/etc/profile.d/conda.sh && conda activate locust'" >> ~/.bashrc
```

验证（计算节点应返回 True；登录节点无卡返回 False 属正常）：

```bash
rocm-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## 安装步骤

### 1. conda 环境

Python 版本须匹配 dtk whl 的 `cpXX` 标记。

```bash
conda create -n locust python=3.10 -y
conda activate locust
```

### 2. 降级 pip

平台预编译的 dtk whl 文件名含两个 `+`，不符合 PEP 440，新版 pip 拒绝安装。

```bash
pip install pip==22.3.1
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 安装 dtk 版 torch

从平台预置路径安装，**不要用 PyPI 公版**（会覆盖导致无法调用 DCU）。

```bash
W=/public/software/apps/DeepLearning/whl/dtk-24.04/pytorch/torch2.1/py310
pip install $W/torch-2.1.0+das1.0+*.whl
pip install $W/torchvision-0.16.0+das1.0+*.whl
```

### 4. 补 OpenSSL 1.1

dtk torch 链接 OpenSSL 1.1，而系统通常是 3.0。装在独立目录，不污染主环境。

```bash
conda create -p $HOME/openssl11 -c conda-forge openssl=1.1.1w -y
```

### 5. 钉住 numpy < 2

dtk torch 用 numpy 1.x 编译，ABI 不兼容 numpy 2.x。

```bash
pip install "numpy==1.26.4"
```

### 6. 安装其余依赖

`--no-deps` 防止 ultralytics 把 dtk torch 换成公版。

```bash
pip install ultralytics --no-deps
pip install matplotlib opencv-python-headless pyyaml scipy tqdm \
    psutil py-cpuinfo ultralytics-thop polars gradio scikit-learn
pip install "pandas==2.2.3"   # 老 GCC 编不了新 pandas，需钉到有预编译 wheel 的版本
```

安装后验证 torch 未被替换（版本号应带 `dtk2404` 后缀）：

```bash
python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
```

## 报错速查

| 报错 | 原因 | 解决 |
|------|------|------|
| `Invalid wheel filename (invalid version)` | 新 pip 拒绝双 `+` 文件名 | 降级 `pip==22.3.1` |
| `libssl.so.1.1: cannot open` | 系统为 OpenSSL 3.0 | 装独立 `openssl=1.1.1w` 并加入 `LD_LIBRARY_PATH` |
| `conflicts with ... dtk/21.10` | 默认加载了旧版 dtk | `module purge` 后重新加载 24.04 |
| `libgalaxyhip.so.5: cannot open` | dtk 版本与 torch 不匹配 | 确保加载 `compiler/dtk/24.04` |
| `NumPy 1.x cannot be run in 2.x` | numpy ABI 不兼容 | `pip install "numpy==1.26.4"` |
| `__has_builtin not detected` | 老 GCC 编不了新 pandas | 装 `pandas==2.2.3` |
| `libmpi.so.40: cannot open` | `module purge` 清掉了 MPI | 补 `module load mpi/hpcx/2.11.0` |
| `ModuleNotFoundError: polars` | ultralytics 8.4 保存模型时需要 | `pip install polars` |
| `ModuleNotFoundError: lap` | ByteTrack 追踪需要 | 首次运行会自动安装，之后重跑一次 |

## 作业提交注意

- 登录节点无加速卡，推理会退回 CPU（慢数十倍），须用 `sbatch` 提交
- `--time` 不可超过账户上限（实测 4 小时），超限报
  `Invalid account or account/partition combination`，报错信息有误导性
- `--mem` 与 `--cpus-per-task` 的比例受 `DefMemPerCPU` 限制，建议省略 `--mem` 用默认值
- Slurm 提交后 python 读取的是**运行时**磁盘上的脚本，修改脚本需重新提交
