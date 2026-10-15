# 默认模型

源码仓库不存放训练权重。可从 [GitHub Releases](https://github.com/dachengac/locust-oviposition/releases)
下载 `locust-oviposition.behavior.zip`，放入本目录，或在软件中选择“导入模型包”。

该包由原项目 [v1.0 的 best.pt](https://github.com/dachengac/locust-oviposition/releases/tag/v1.0)
转换得到。原 PT SHA-256：

`56e3f750492ff16927142ff01e1545a82b073e1e4435c73e824c0c9475b78de8`

输入为固定的 RGB NCHW float32 `[1,3,640,640]`，输出 `[1,5,8400]`，类别为 `locust`。
模型负责检测飞蝗；产卵候选由包中的净位移时序规则计算。
替换检测器不代表自动学习新的动作分类，接口与规则需一并匹配。

包接口见 [MODEL_PACK.md](../docs/MODEL_PACK.md)，验证记录见 [VALIDATION.md](../docs/VALIDATION.md)。
