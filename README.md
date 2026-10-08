# 激光原理与技术（I）课程大作业

作者：彭友  
班级：光电2404班  
学号：U202413812

本项目完成题目二：对称共焦腔的一维、二维 Fox–Li 自再现模式计算，以及单面反射镜倾斜分析。Python 负责迭代和绘图，报告见 [报告.md](报告.md)。

## 环境构建

使用 Python 3.10 或更新版本。打开终端，进入包含本 README 的项目根目录。


自行创建虚拟环境，激活后安装依赖：

```bash
python -m pip install -r requirement.txt
```

依赖为 NumPy、Matplotlib 和 PyYAML。

## 代码启动方式

以下命令均在项目根目录执行。

运行一维和二维基础仿真：

```bash
python src/Python/run_baseline.py --mode both
```

也可以分别运行：

```bash
python src/Python/run_baseline.py --mode 1d
python src/Python/run_baseline.py --mode 2d
```

运行单镜倾斜扫描：

```bash
python src/Python/run_tilt.py
```

程序会完成计算并保存图像，结束时打印结果目录。倾斜扫描包含多个角度，部分角度可能执行到最大迭代次数，运行时间会比基础仿真长。

## 参数设置

所有可调参数集中在 [parameter.yaml](src/Python/config/parameter.yaml)，修改后重新运行程序。

| 配置项 | 内容 |
|---|---|
| `project` | 波长、腔长、两面镜子的曲率半径 |
| `grid` | 一维/二维网格点数、窗口和孔径倍数、FFT 补零倍数 |
| `fox_li` | 最大迭代次数、误差阈值、连续稳定轮数、初场和随机种子 |
| `tilt` | 倾斜镜面、扫描角度和初场延续设置 |
| `output` | 结果输出目录 |

长度使用 m，倾角列表使用 mrad。当前模型要求 `mirror1_radius_m = mirror2_radius_m = cavity_length_m`，即对称共焦条件。窗口和孔径倍数以共焦镜面光斑半径为基准。命令行的 `--mode` 只选择计算维度。

## 文件结构

```text
.
├── README.md                  # 环境和运行说明
├── 报告.md                    # 推导、仿真结果和分析
├── agents.md                  # 项目协作约定
├── requirement.txt            # Python 依赖
├── assert/                    # 报告使用的 11 张 PNG 和示意图 SVG 源文件
│   └── manifest.json          # 图片与原始数据的对应关系
├── doc/                       # 课程作业要求
├── src/Python/                # 仿真与绘图代码
│   ├── config/parameter.yaml  # 唯一可调参数文件
│   ├── run_baseline.py        # 基础仿真入口
│   └── run_tilt.py            # 倾角扫描入口
└── results/
    ├── manifest.json          # 当前结果索引
    └── runs/                  # 报告依据的数据及后续新运行结果
```

## 查看结果

每次运行创建新的 `results/runs/<UTC时间戳>/` 目录，并保存当次参数快照 `parameter_used.yaml`。基础结果在 `baseline/1d/` 和 `baseline/2d/`，倾斜结果在 `tilt/`。

| 文件 | 内容 |
|---|---|
| `summary.json` | 本征值、损耗、迭代次数、收敛状态和光斑指标 |
| `history.csv` | 每轮复场差、光强差和本征值变化 |
| `fields.npz` / `fields_theta_*mrad.npz` | 网格坐标和两镜面的复光场 |
| `scan.csv` | 各倾角的定量结果 |
| `*.png` | Python 生成的场分布和收敛图 |

`converged: false` 表示未满足收敛条件，最后一轮的场不能作为已经收敛的模式。
