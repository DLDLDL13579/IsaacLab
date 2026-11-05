# Isaac Lab 2.3.2 上游副本：GPU 加速机器人学习仿真框架（本地学习与实验用）

**一句话定位**：[NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) 开源框架的**完整副本**，构建在 [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) 之上，用 GPU 加速把强化学习、模仿学习、运动规划等机器人研究流程统一起来，适合做 sim-to-real 迁移。

> 🔴 **非原创声明（第一屏必读）**
>
> 本仓库是 NVIDIA / The Isaac Lab Project Developers 的开源项目副本，**用于本地学习与实验，并非原创项目**。代码与文档的绝大部分权利归**原作者及各自上游组件权利人**所有。本副本相对上游**仅新增根目录 `check_x30.py`** 一个调试脚本（该说法与 `docker/` 下的实际改动有冲突，见[已知问题](#已知问题)）。

| 项 | 内容 |
|---|---|
| 上游仓库 | https://github.com/isaac-sim/IsaacLab |
| 版本 | **2.3.2**（现有 README 称见 `VERSION`；已用完整 `git ls-tree` 复核该文件确实存在，文件值未逐字展开） |
| 版权 | Copyright (c) 2022-2025, **The Isaac Lab Project Developers**（`LICENSE` 原文） |
| 许可证 | **BSD-3-Clause**（`LICENSE`）+ **Apache-2.0**（`LICENSE-mimic`，`isaaclab_mimic` 扩展及其独立脚本）；依赖与资产许可见 [`docs/licenses`](docs/licenses) |
| 引用 | 用于研究请按 `CITATION.cff` 引用上游技术报告（BibTeX 见[引用](#引用)） |

**在体系里的位置**：本账号下另有三套机器人相关仓库，但它们与本副本**没有代码依赖**，仅作导航：[`stm32-robot-base`](https://github.com/DLDLDL13579/stm32-robot-base)（四轮差速底盘 STM32 固件）、[`stm32-ros2-navigation`](https://github.com/DLDLDL13579/stm32-ros2-navigation)（ROS 2 上位机建图）、[`upboard-robot-software`](https://github.com/DLDLDL13579/upboard-robot-software)（四足控制软件固件备份）。

---

## 目录

- [现状与状态表](#现状与状态表)
- [目录结构](#目录结构)
- [关键能力](#关键能力)
- [关键实现与代码依据](#关键实现与代码依据)
- [参数表](#参数表)
- [快速开始](#快速开始)
- [Isaac Sim 版本依赖](#isaac-sim-版本依赖)
- [文档构建](#文档构建)
- [Docker 与集群部署](#docker-与集群部署)
- [上游社区与贡献](#上游社区与贡献)
- [排障与支持](#排障与支持)
- [已知问题](#已知问题)
- [归属与许可证](#归属与许可证)
- [引用](#引用)
- [致谢](#致谢)

---

## 现状与状态表

| 项 | 内容 | 出处 |
|---|---|---|
| 仓库 | `DLDLDL13579/IsaacLab`，公开，默认分支 `main` | GitHub 元数据 |
| 主语言 / 体积 | Python / 36791 KB（约 36 MB） | GitHub 元数据 |
| star / fork | 0 / 0 | GitHub 元数据 |
| 创建 / 最后 push | 2026-08-05 / **2026-09-17** | GitHub 元数据 |
| GitHub 描述 | **空** | GitHub 元数据 |
| 文件规模 | 共 **2030** 个文件；一级分布 `source:1144`、`docs:605`、`scripts:153`、`tools:40`、`docker:24`、`.github:21`、`apps:21`、根文件 `20`、`.vscode:2` | GitHub 文件树一级统计 |
| 项目性质 | 第三方上游副本 + 本地调试脚本；**不是本人原创项目，也不是 Isaac Lab 官方仓库** | 现有 README 副本声明 |
| LICENSE | 根目录有 `LICENSE` 与 `LICENSE-mimic` **两个**许可文件 | GitHub 元数据 |
| 已完成 | 上游 2.3.2 的 `source/`、`docs/`、`scripts/`、`tools/`、`docker/`、`.github/` 等主体目录在副本内可见；新增 `check_x30.py`；许可文件齐备 | 现有 README + 文件树一级统计 |
| 未完成 / 受限 | `tests/` 被 `.gitignore` 全局忽略；USD 资产不允许入库；Isaac Sim 本体（含专有许可组件）**不在本仓库** | `.gitignore` 原文 + 现有 README License 节 |
| 官方 CI 状态 | 徽章指向**上游** `isaac-sim/IsaacLab` 的 workflow，**不代表本副本跑过 CI** | 现有 README 徽章 URL |

> 事实包的文件树是**节选**：一级统计 2030 个文件，但清单只展开约 170 行路径，`source/` 的 1144 个文件在清单里**一个都没有展开**。因此本文的目录结构与数量结论**不保证穷尽**。
> 已用完整 `git ls-tree` 复核若干关键路径：`check_x30.py`、`VERSION`、`CITATION.cff`、`docker/cluster/run_singularity.sh` **均确实存在**；凡事实包节选未覆盖、且未另行复核的，仍标注为「事实包无法核实」，**不要据此断言文件不存在**。

---

## 目录结构

只列事实包清单中真实存在的路径。凡标注「清单未展开」的，是由本仓库其它文件（README、`docs/README.md`、`docker-compose.yaml`、`run_singularity.sh`）明确引用、但文件树清单未展开的路径，**不能当作已核实存在**。

```
IsaacLab/
├── LICENSE                              # BSD-3-Clause，Copyright (c) 2022-2025
├── LICENSE-mimic                         # Apache-2.0（isaaclab_mimic 扩展）
├── README.md                             # 中文副本声明 + 上游原始 README
├── CONTRIBUTING.md                       # 贡献指南 + Developer Certificate of Origin 1.1
├── CONTRIBUTORS.md                       # 贡献者名单
├── SECURITY.md                           # 安全政策
├── check_x30.py                          # ★ 副本声明称唯一新增：检查 X30.usd 结构
├── .gitignore                            # 忽略构建产物、USD、输出、数据集、tests/ 等
├── .pre-commit-config.yaml               # ruff / codespell / insert-license 等钩子
├── .github/
│   ├── ISSUE_TEMPLATE/{bug,proposal,question}.md
│   ├── LICENSE_HEADER.txt                # BSD-3-Clause 头，insert-license 用
│   ├── LICENSE_HEADER_MIMIC.txt          # Apache-2.0 头，mimic 文件专用
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── actions/{combine-results,docker-build,run-tests}/action.yml
│   ├── labeler.yml / stale.yml
│   └── workflows/                        # 9 个文件（见下方重数）
│       ├── build.yml / check-links.yml / daily-compatibility.yml / docs.yaml
│       ├── labeler.yml / license-check.yaml / license-exceptions.json
│       └── postmerge-ci.yml / pre-commit.yaml
├── .vscode/                              # 2 个文件，清单未展开
├── apps/
│   ├── isaacsim_4_5/extension.toml       # Isaac Sim 4.5 应用配置
│   └── rendering_modes/extension.toml    # 渲染模式配置
├── docker/
│   ├── container.py / container.sh       # 容器生命周期入口
│   ├── Dockerfile.base / Dockerfile.ros2 # 由 compose 引用，清单未展开
│   ├── docker-compose.yaml               # base / ros2 两个 profile
│   ├── docker-compose.cloudxr-runtime.patch.yaml  # CloudXR runtime 补丁
│   ├── custom_mount.yaml / x11.yaml
│   ├── .ros/{cyclonedds.xml,fastdds.xml}# ROS 2 DDS 配置
│   ├── .env.base / .env.ros2             # 由 compose env_file 引用（清单未展开）
│   ├── cluster/                          # 4 个脚本
│   │   ├── cluster_interface.sh
│   │   ├── run_singularity.sh            # Singularity 计算节点启动
│   │   ├── submit_job_pbs.sh
│   │   └── submit_job_slurm.sh
│   ├── test/test_docker.py               # 4 组 profile/suffix 起停测试
│   ├── utils/{__init__,container_interface,state_file,x11_utils}.py
│   └── .ros/…                            # 见上
├── docs/
│   ├── Makefile                          # current-docs / multi-docs 两个目标
│   ├── README.md                         # Sphinx 文档构建（Linux / Windows）
│   ├── conf.py                           # Sphinx 配置
│   ├── requirements.txt                  # docs/README.md 中 pip install -r 引用，清单未展开
│   ├── source/_static/isaaclab.jpg       # 现有 README 首图引用
│   ├── _redirect/index.html              # 多版本文档跳转页
│   ├── _templates/versioning.html
│   └── licenses/
│       ├── assets/                       # 6 个资产许可：anymal_b、anymal_c、franka、
│       │                                 #   kinova、robotiq、unitree
│       └── dependencies/                 # 大量依赖许可文本（isaacsim、cuRobo、
│                                         #   legged-gym、onnx、numpy、… 部分展开）
├── source/                               # 1144 个文件，清单未展开
├── scripts/                              # 153 个文件，清单未展开
├── tools/                                # 40 个文件，清单未展开
└── （根目录 20 个文件中，清单另约 12 个未展开）
```

**`.github/` 重数**（清单逐条）：`ISSUE_TEMPLATE` 3 + `LICENSE_HEADER.txt`、`LICENSE_HEADER_MIMIC.txt`、`PULL_REQUEST_TEMPLATE.md` 3 + `actions` 3 + `labeler.yml`、`stale.yml` 2 = **11 个**；`workflows/` 下 `build.yml`、`check-links.yml`、`daily-compatibility.yml`、`docs.yaml`、`labeler.yml`、`license-check.yaml`、`license-exceptions.json`、`postmerge-ci.yml`、`pre-commit.yaml` = **9 个**（其中 `.yml`/`.yaml` workflow **8 个**，`license-exceptions.json` 是配置数据不是 workflow）。合计清单展开 **20 个**，一级统计 `.github:21`，尚有 **1 个未展开**。

**`docker/` 重数**：`container.py`、`container.sh`、`custom_mount.yaml`、`docker-compose.cloudxr-runtime.patch.yaml`、`docker-compose.yaml`、`test/test_docker.py`、`utils/__init__.py`、`container_interface.py`、`state_file.py`、`x11_utils.py`、`x11.yaml`、`cluster_interface.sh`、`run_singularity.sh`、`submit_job_pbs.sh`、`submit_job_slurm.sh`、`.ros/cyclonedds.xml`、`.ros/fastdds.xml` = **17 个**；一级统计 24，余 **7 个未在清单展开**。其中 `Dockerfile.base`、`Dockerfile.ros2`、`.env.base`、`.env.ros2` 是由 `docker-compose.yaml` 与 `run_singularity.sh` 明确引用但清单未列出的文件，其余未展开文件不详。

> `source/`、`scripts/`、`tools/` 三个主体目录合计 1337 个文件在事实包清单中**完全没有展开**，所以本文不列举具体 Python 包名、环境名或脚本路径——那些必须回上游仓库或本地源码确认。

---

## 关键能力

以下内容对应上游 README 的 **Key Features**，中文转述，术语与专有名词保持原文；原文中的数量断言（`more than 16` / `more than 30`）标注为上游自述，事实包文件树为节选，无法重数核实。

**Isaac Lab** 是一个 GPU 加速的开源框架，目标是统一并简化机器人研究工作流（强化学习、模仿学习、运动规划）。它构建在 NVIDIA Isaac Sim 之上，把快速且准确的物理与传感器仿真结合起来，是 sim-to-real 迁移的合适选择。框架提供精确的传感器仿真（基于 RTX 的相机、LIDAR、接触传感器等）；GPU 加速让用户能更快地跑复杂仿真与计算，这对强化学习这类迭代式、数据密集型任务尤为关键；同时既可本地运行，也可分布式部署到云端。详细描述见上游 [arXiv paper](https://arxiv.org/abs/2511.04831)。

| 类别 | 上游 README 声明的能力 |
|---|---|
| Robots | 覆盖 manipulators、quadrupeds、humanoids，上游自述 **more than 16** 个常用模型 |
| Environments | 上游自述 **more than 30** 个可开箱训练的环境，支持 RSL RL、SKRL、RL Games、Stable Baselines 等 RL 框架，并支持 multi-agent reinforcement learning |
| Physics | Rigid bodies、articulated systems、deformable objects |
| Sensors | RGB / depth / segmentation cameras、camera annotations、IMU、contact sensors、ray casters |

**徽章所声明的基线**（现有 README 原文，链接均指向上游）：

| 徽章 | 值 | 链接 |
|---|---|---|
| IsaacSim | 5.1.0 | https://docs.isaacsim.omniverse.nvidia.com/latest/index.html |
| Python | 3.11 | https://docs.python.org/3/whatsnew/3.11.html |
| Linux platform | linux-64 | https://releases.ubuntu.com/22.04/ |
| Windows platform | windows-64 | https://www.microsoft.com/en-us/ |
| pre-commit | 上游 `pre-commit.yaml` workflow 状态 | https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml |
| docs status | 上游 `docs.yaml` workflow 状态 | https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml |
| License | BSD-3-Clause | https://opensource.org/licenses/BSD-3-Clause |
| License | Apache-2.0 | https://opensource.org/license/apache-2-0 |

> 这些徽章是**上游 README 原样**，本副本 GitHub 描述为空、star/fork 为 0，且 CI 链接指向 `isaac-sim` 而非 `DLDLDL13579`，不能据此认为本副本通过了 CI。

---

## 关键实现与代码依据

由于 `source/` 未展开，本节只讲事实包里**逐字给出原文**的四个机制，说明它们为什么这样设计。

### 1. 双许可证通过 pre-commit 自动插入文件头

`.pre-commit-config.yaml` 里配置了**两个 `insert-license` 钩子**，用不同 `LICENSE_HEADER` 文件与不同路径范围实现「主体 BSD-3 / mimic Apache-2.0」：

```yaml
  - repo: https://github.com/Lucas-C/pre-commit-hooks
    rev: v1.5.5
    hooks:
      - id: insert-license
        files: \.(py|ya?ml)$
        args:
          - --license-filepath
          - .github/LICENSE_HEADER.txt
          - --use-current-year
        exclude: "source/isaaclab_mimic/|scripts/imitation_learning/isaaclab_mimic/"
  # Apache 2.0 license for mimic files
  - repo: https://github.com/Lucas-C/pre-commit-hooks
    rev: v1.5.5
    hooks:
      - id: insert-license
        files: ^(source/isaaclab_mimic|scripts/imitation_learning/isaaclab_mimic)/.*\.py$
        args:
          - --license-filepath
          - .github/LICENSE_HEADER_MIMIC.txt
          - --use-current-year
```

**为什么这么设计**：主体代码与 `isaaclab_mimic` 扩展授权不同，靠人工维护文件头必然漂移。第一个钩子用 `exclude` 排除 mimic 路径，第二个钩子用 `files` 正则只圈住 mimic 路径，两者互补且不重叠；`--use-current-year` 让版权年份随提交年份自动增长（这也解释了为什么 `.pre-commit-config.yaml` / `docker/*.yaml` 的头是 `2022-2026`，而 `LICENSE` 是 `2022-2025`）。同一配置里 `check-added-large-files --maxkb=2000` 把超过 2 MB 的新增文件挡在门外，注释明确写着 `Should use git-lfs instead`。

### 2. Docker 的扩展字段复用，把缓存与源码分离

`docker/docker-compose.yaml` 用 YAML 扩展字段（`&` / `*`）定义三份可复用片段，再让 `isaac-lab-base` 与 `isaac-lab-ros2` 两个 service 引用：

```yaml
x-default-isaac-lab-volumes: &default-isaac-lab-volumes
  - type: volume
    source: isaac-cache-kit
    target: ${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache
  ...
x-default-isaac-lab-environment: &default-isaac-lab-environment
  - ISAACSIM_PATH=${DOCKER_ISAACLAB_PATH}/_isaac_sim
  - OMNI_KIT_ALLOW_ROOT=1
  - ISAAC_NUCLEUS_DIR=/isaac_offline_assets
x-default-isaac-lab-deploy: &default-isaac-lab-deploy
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [ gpu ]
```

**为什么这么设计**（文件里的注释写得很直白）：
- 大量 `volume` 挂载 Isaac Sim 的 cache / logs / data / Documents，避免容器反复重建时重新下载或编译着色器缓存；
- `../source`、`../scripts`、`../docs`、`../tools` 用 **bind** 挂载，使宿主机改代码立刻在容器内生效；
- 同时把 `docs/_build`、`logs`、`data_storage` 放进 volume，注释说这样能「防止 root 属主文件淹没宿主机的 `_build` 和 logs 目录」，并「把产物留在持久 volume 里便于回拷」；
- `deploy.resources.reservations.devices` 用 `nvidia` driver + `count: all` + `capabilities: [gpu]` 申请全部 GPU。

两个 service 的差异只有 `profiles`（`base` / `ros2`）、`env_file`（ros2 多一个 `.env.ros2`）和 Dockerfile（`Dockerfile.base` / `Dockerfile.ros2`），其余通过 `*default-isaac-lab-*` 复用。`isaac-lab-ros2` 里 `ROS2_APT_PACKAGE=${ROS2_APT_PACKAGE:-NONE}` 的注释说明：默认 `NONE` 是为了「只构建 base profile 时不出警告」。

### 3. 集群作业把整个 Isaac Lab 目录同步到计算节点

`docker/cluster/run_singularity.sh` 的做法是：先 `source` 集群环境变量 → `setup_directories()` 建齐 8 个缓存子目录，另 `mkdir -p` 一个 `$CLUSTER_ISAACLAB_DIR/logs` → `cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR` 与 `cp -r $1 $TMPDIR` 把缓存和最新代码整份搬到计算节点 → `tar -xf $CLUSTER_SIF_PATH/$2.tar` 解出容器 → `singularity exec` 带 **10 个 `-B` 绑定**（8 个缓存 + 代码目录 + logs） + `--nv --writable --containall` 启动 → 结束后 `rsync -azPv` 把缓存回写。

```bash
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    ... \
    -B $TMPDIR/$dir_name:/workspace/isaaclab:rw \
    --nv --writable --containall $TMPDIR/$2.sif \
    bash -c "export ISAACLAB_PATH=/workspace/isaaclab && cd /workspace/isaaclab && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:3}"
```

脚本注释解释了关键一步：正常情况 `ISAACLAB_PATH` 由 `isaaclab.sh` 设置，但这里因为「同步了整个 Isaac Lab 目录并移除了指向 isaac-sim 的符号链接」，所以直接调 isaac-sim 的 `python.sh` 并手动 export。

### 4. 文档双版本构建

`docs/Makefile` 只有两个目标，`SOURCEDIR = .`、`BUILDDIR = _build`、`SPHINXBUILD ?= sphinx-build`：

```make
.PHONY: multi-docs
multi-docs:
	@sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@cp _redirect/index.html $(BUILDDIR)/index.html

.PHONY: current-docs
current-docs:
	@rm -rf "$(BUILDDIR)/current"
	@$(SPHINXBUILD) -W --keep-going "$(SOURCEDIR)" "$(BUILDDIR)/current" $(SPHINXOPTS)
```

**为什么这么设计**：`current-docs` 用 `-W --keep-going` 把 Sphinx 警告升级为错误但仍尽量跑完全部文件，适合 CI 卡质量；`multi-docs` 用 `sphinx-multiversion` 生成历史 tag + main 的多版本站点，再盖一个 `_redirect/index.html` 做入口跳转。

### 5. Docker 起停的参数化测试

`docker/test/test_docker.py` 用 `pytest.mark.parametrize` 跑 4 组：

```python
("base", ""), ("base", "test"), ("ros2", ""), ("ros2", "test")
```

`start_stop_docker()` 的完整链路是 `python container.py start <profile> [--suffix X]` → `docker ps` 里断言容器名出现 → `python container.py stop ...` → 再断言容器名消失。容器名规则为 `isaac-lab-{profile}` 或 `isaac-lab-{profile}-{suffix}`。

---

## 参数表

以下数值与命令逐字摘自事实包给出的文件原文；未标注单位者代码未给出单位。

### 环境与版本

| 参数 | 值 | 出处 |
|---|---|---|
| Python | 3.11 | 现有 README 徽章 |
| Isaac Sim（徽章） | 5.1.0 | 现有 README 徽章 |
| Isaac Sim（`v2.3.X` 兼容） | 4.5 / 5.0 / 5.1 | 现有 README 版本依赖表 |
| 平台 | linux-64 / windows-64 | 现有 README 徽章 |
| `SOURCEDIR` | `.` | `docs/Makefile` |
| `BUILDDIR` | `_build` | `docs/Makefile` |
| `SPHINXBUILD` | `sphinx-build` | `docs/Makefile` |
| `SPHINXOPTS` | 空（可由命令行 / 环境覆盖） | `docs/Makefile` |

### pre-commit 钩子

| 钩子 / 仓库 | 版本 / 参数 | 出处 |
|---|---|---|
| `ruff-pre-commit`（`ruff`、`ruff-format`） | `rev: v0.14.10`；`args: ["--fix"]` | `.pre-commit-config.yaml` |
| `pre-commit/pre-commit-hooks` | `rev: v6.0.0` | 同上 |
| `check-added-large-files` | `args: ["--maxkb=2000"]`（2 MB） | 同上 |
| `codespell` | `rev: v2.4.1`，`additional_dependencies: [tomli]`，exclude 两个路径：`CONTRIBUTORS.md` 与 `docs/source/setup/walkthrough/concepts_env_design.rst` | 同上 |
| `Lucas-C/pre-commit-hooks` (`insert-license` ×2) | `rev: v1.5.5` | 同上 |
| `pre-commit/pygrep-hooks` | `rev: v1.10.0`；钩子 `rst-backticks`、`rst-directive-colons`、`rst-inline-touching-normal` | 同上 |
| 许可证头 | `.github/LICENSE_HEADER.txt`（BSD-3）/ `.github/LICENSE_HEADER_MIMIC.txt`（Apache-2.0） | 同上 |
| `pyright` | **已注释停用**，注释原因 `FIXME: Figure out why this is getting stuck under VPN.` | 同上 |

### Docker / CloudXR

| 参数 | 值 | 出处 |
|---|---|---|
| `OMNI_KIT_ALLOW_ROOT` | `1` | `docker/docker-compose.yaml` |
| `ISAAC_NUCLEUS_DIR` | `/isaac_offline_assets` | `docker/docker-compose.yaml` |
| `ISAACSIM_PATH` | `${DOCKER_ISAACLAB_PATH}/_isaac_sim` | 同上 |
| 离线资产 bind | `source: /home/gkfd/isaac_offline_assets` → `target: /isaac_offline_assets` | 同上 |
| GPU 申请 | `driver: nvidia`、`count: all`、`capabilities: [gpu]` | 同上 |
| `network_mode` | `host` | 同上（两个 service） |
| 容器名 / 镜像 | `isaac-lab-base${DOCKER_NAME_SUFFIX-}`、`isaac-lab-ros2${DOCKER_NAME_SUFFIX-}` | 同上 |
| `ROS2_APT_PACKAGE` 默认 | `${ROS2_APT_PACKAGE:-NONE}` | 同上 |
| CloudXR 端口（TCP） | `48010` | `docker/docker-compose.cloudxr-runtime.patch.yaml` |
| CloudXR 端口（UDP） | `47998`、`47999`、`48000`、`48005`、`48008`、`48012` | 同上 |
| CloudXR healthcheck | `test: ["CMD", "test", "-S", "/openxr/run/ipc_cloudxr"]`、`interval: 1s`、`timeout: 1s`、`retries: 10`、`start_period: 5s` | 同上 |
| `XDG_RUNTIME_DIR` / `XR_RUNTIME_JSON` | `/openxr/run` / `/openxr/share/openxr/1/openxr_cloudxr.json` | 同上 |
| 测试矩阵 | `base` / `ros2` × suffix `""` / `"test"` | `docker/test/test_docker.py` |

### 集群脚本

| 参数 | 值 | 出处 |
|---|---|---|
| 缓存子目录（`setup_directories()` 建 8 个） | `cache/kit`、`cache/ov`、`cache/pip`、`cache/glcache`、`cache/computecache`、`logs`、`data`、`documents`；另 `mkdir -p $CLUSTER_ISAACLAB_DIR/logs` | `docker/cluster/run_singularity.sh` |
| 环境变量文件 | `$SCRIPT_DIR/.env.cluster`、`$SCRIPT_DIR/../.env.base` | 同上 |
| singularity 参数 | `--nv --writable --containall` | 同上 |
| 集群内 `ISAACLAB_PATH` | `/workspace/isaaclab` | 同上 |
| Python 解释器 | `/isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE}` | 同上 |

---

## 快速开始

**前置条件**：Python 3.11（徽章）；与版本匹配的 **Isaac Sim 4.5 / 5.0 / 5.1**（见[版本依赖](#isaac-sim-版本依赖)）；Linux 64 位或 Windows 64 位；GPU 与对应驱动（Docker 路径需 NVIDIA Container Toolkit）；Isaac Sim 本体**不在本仓库**，需另行安装。

1. **克隆本副本**（或直接用上游仓库）

```bash
git clone https://github.com/DLDLDL13579/IsaacLab.git
cd IsaacLab
```

> 若目标是跟上官方更新，应优先克隆 https://github.com/isaac-sim/IsaacLab —— 本副本最后 push 为 2026-09-17，且 CI 徽章指向上游。

2. **按官方文档安装**

上游 [documentation page](https://isaac-sim.github.io/IsaacLab) 提供全部上手内容，现有 README 列出的入口：

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)

3. **本地构建文档**（可选，命令逐字来自 `docs/README.md`）

```bash
cd docs
pip install -r requirements.txt
make current-docs
xdg-open _build/current/index.html
```

> `docs/README.md` **强烈建议用 Python 虚拟环境**隔离依赖，避免与系统全局 Python 冲突。Windows 侧文档给出的是 `start _build\current\index.html`，但仍用 `make current-docs`——见[已知问题](#已知问题)。

4. **Docker 路径**（命令依据 `docker/test/test_docker.py` 的调用方式）

```bash
cd docker
python container.py start base
python container.py stop base

python container.py start ros2 --suffix test
python container.py stop ros2 --suffix test
```

`docker/test/test_docker.py` 用 `pytest.mark.parametrize` 组织这四组起停测试（文件内有 `@pytest` 与 `subprocess` 依赖），可在仓库根目录用 `pytest docker/test/test_docker.py` 验证容器名与 `docker ps` 输出；该命令是依据 pytest 常规用法给出，事实包未提供 CI 里的实际调用命令。

5. **LCM / 组播与集群**：本副本未给出相关命令，集群入口为 `docker/cluster/run_singularity.sh`、`submit_job_pbs.sh`、`submit_job_slurm.sh`，需自备 `.env.cluster` 与 `.env.base`（见[已知问题](#已知问题)）。

---

## Isaac Sim 版本依赖

Isaac Lab 构建在 Isaac Sim 之上，各 release / 分支要求特定版本的 Isaac Sim。以下表逐字保留上游 README 的对应关系：

| Isaac Lab Version | Isaac Sim Version |
| --- | --- |
| `main` branch | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.X` | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.X` | Isaac Sim 4.5 / 5.0 |
| `v2.1.X` | Isaac Sim 4.5 |
| `v2.0.X` | Isaac Sim 4.5 |

本副本现有 README 自述版本 **2.3.2**，按上表应对应 Isaac Sim 4.5 / 5.0 / 5.1；徽章另标 5.1.0。

---

## 文档构建

`docs/README.md` 原文要点：使用 [Sphinx](https://www.sphinx-doc.org/en/master/) 与 [Book Theme](https://sphinx-book-theme.readthedocs.io/en/stable/) 维护生成文档。

| 目标 | Linux 命令 | Windows 打开方式 | 说明 |
|---|---|---|---|
| 当前版本文档 | `cd docs` → `pip install -r requirements.txt` → `make current-docs` | `xdg-open _build/current/index.html` / `start _build\current\index.html` | 构建当前版本，输出到 `_build/current` |
| 多版本文档 | `cd docs` → `pip install -r requirements.txt` → `make multi-docs` | `xdg-open _build/index.html` / `start _build\index.html` | 含历史 tag 与 main 分支，入口由 `_redirect/index.html` 覆盖 |

对应 `docs/Makefile` 的 `current-docs` 与 `multi-docs` 两个 `.PHONY` 目标（见[关键实现](#关键实现与代码依据)）。

---

## Docker 与集群部署

- `docker/docker-compose.yaml` 定义 `isaac-lab-base`（`profiles: ["base"]`）与 `isaac-lab-ros2`（`profiles: ["ros2"]`，在 base 之上加装 **ROS2 Humble**）两个服务，均使用 `network_mode: host`、`entrypoint: bash`、`stdin_open: true`、`tty: true`。
- `docker/docker-compose.cloudxr-runtime.patch.yaml` 追加 `cloudxr-runtime` 服务，暴露 1 个 TCP + 6 个 UDP 端口，并通过 `openxr-volume` 与 `isaac-lab-base` 共享 `/openxr`；healthcheck 用 socket 文件 `/openxr/run/ipc_cloudxr` 判定就绪。
- `docker/.ros/cyclonedds.xml` 与 `docker/.ros/fastdds.xml` 是 ROS 2 的 DDS 配置（内容未在事实包展开）。
- `docker/cluster/run_singularity.sh` 面向 Singularity 集群；`submit_job_pbs.sh` 与 `submit_job_slurm.sh` 分别对应 PBS / Slurm 作业提交（内容未展开）。
- `docker/test/test_docker.py` 覆盖 4 组 profile / suffix 组合的起停。

---

## 上游社区与贡献

以下为上游 README 原意的中文转述，链接保持原文：

- **Contributing to Isaac Lab**：上游欢迎社区的 bug 报告、功能请求与代码贡献，细节见 [contribution guidelines](https://isaac-sim.github.io/IsaacLab/main/source/refs/contributing.html)。本仓库 `CONTRIBUTING.md` 全文给出 **Developer Certificate of Origin 1.1**（Copyright (C) 2004, 2006 The Linux Foundation and its contributors），规定贡献者须证明 (a) 自己创作并有权按文件所示开源许可提交、(b) 基于他人已有工作且该工作受适当开源许可覆盖、(c) 由他人直接提供且自己未修改，并 (d) 理解贡献与相关个人信息会被公开且长期保留。
- **Show & Tell: Share Your Inspiration**：鼓励在 [Show & Tell](https://github.com/isaac-sim/IsaacLab/discussions/categories/show-and-tell) 分享自建教程、学习内容与项目，以启发他人、促成协作。
- **Connect with the NVIDIA Omniverse Community**：有项目想推广可联系 `OmniverseCommunity@nvidia.com`；也可加入 [Omniverse Discord](https://discord.com/invite/nvidiaomniverse) 与开发者交流、分享项目。

---

## 排障与支持

上游 README 的 **Troubleshooting** 与 **Support** 两节：

- 常见问题先看 [troubleshooting](https://isaac-sim.github.io/IsaacLab/main/source/refs/troubleshooting.html)，或 [submit an issue](https://github.com/isaac-sim/IsaacLab/issues)。
- 若问题出在 **Isaac Sim** 本体，建议查其 [documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html) 或在 [NVIDIA forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67) 提问。
- **Discussions** 用于讨论想法、提问、请求新功能；**Issues** 只用于跟踪范围明确、交付物清晰的可执行工作（修 bug、文档问题、新功能、一般性更新）。

> 以上链接全部指向上游 `isaac-sim/IsaacLab`，**不是本副本**。在本副本里开 issue 不会得到上游维护者响应。

---

## 已知问题

| # | 问题 | 证据 | 影响 / 建议 |
|---|---|---|---|
| 1 | **文档描述落后于代码**：副本声明写「本副本相对上游的改动：仅新增 `check_x30.py`…其余代码与文档均为上游原样」，但 `docker/docker-compose.yaml` 中存在本地绝对路径 `source: /home/gkfd/isaac_offline_assets` → `target: /isaac_offline_assets`，且环境变量 `ISAAC_NUCLEUS_DIR=/isaac_offline_assets` 指向该离线资产目录 | 现有 README 副本声明 vs `docker/docker-compose.yaml` 逐字原文 | 至少 `docker/` 下有本地定制（离线 Nucleus 资产）。**以代码为准**：改声明为「另含离线资产挂载定制」，或把该路径改回上游形式；否则 Clone 者在自己机器上会因路径不存在而挂载失败 |
| 2 | **「仅新增一个文件」的说法无法核实**：事实包文件树一级统计 2030 个文件，`source/`(1144)、`scripts/`(153)、`tools/`(40) 三大目录**一个文件都没展开** | 文件树一级分布 + 清单 | 无法逐字比对上游来证明「其余原样」。本 README 保留该说法作为**副本作者自述**，不作为已核实事实 |
| 3 | **版本号依据文件未在事实包展开**：声明「版本 2.3.2（见 `VERSION`）」，但清单展开的根文件中没有 `VERSION`；同时 `LICENSE` 头为 `2022-2025`，而 `.pre-commit-config.yaml` 与 `docker/*.yaml` 的头为 `2022-2026` | 现有 README + `LICENSE` + `.pre-commit-config.yaml` + `docker/docker-compose.yaml` 头部 | 版本号请以本地 `VERSION` 文件为准（本 README 按现有 README 转述为 2.3.2）；年份差异是 `--use-current-year` 自动插入文件头造成的，属正常，但对外引用版权年份时以 `LICENSE` 为准 |
| 4 | **GitHub 描述为空**，副本身份只在 README 里说明；CI 徽章指向 `isaac-sim/IsaacLab` 而非本仓库 | GitHub 元数据 + 现有 README 徽章 URL | 从仓库列表 / API 看不出这是副本，易被误认作原创或官方镜像；建议补描述并注明上游地址 |
| 5 | **`check_x30.py` 的目标文件被 `.gitignore` 全局排除**：仓库禁止 `**/*.usd`、`**/*.usda`、`**/*.usdc`、`**/*.usdz` 入库（注释 `# No USD files allowed in the repo`），而该脚本是用于检查 `X30.usd` 文件结构的 | `.gitignore` + 现有 README 副本声明 | `X30.usd` 永远不会在仓库里，脚本只能在本地 Isaac Sim 资产目录下手工指向该文件；脚本本身内容未在事实包展开，具体用法需读源码 |
| 6 | **`.gitignore` 全局忽略 `tests/`**：`tests/` 一行无路径限定，会忽略任意层级同名目录 | `.gitignore` 原文 | 本地新增测试目录不会进版本库；`docker/test/test_docker.py` 位于 `docker/test/`（复数不同）不受影响，但仍建议确认上游是否有被误忽略的测试 |
| 7 | **`run_singularity.sh` 把变量当命令执行**：结尾 `if $REMOVE_CODE_COPY_AFTER_JOB; then rm -rf $1; fi` —— 该变量未在本脚本内定义，也未做 `:=` 兜底；若 `.env.cluster` / `.env.base` 未定义它，shell 会尝试执行空命令或把值当命令名 | `docker/cluster/run_singularity.sh` 逐字原文 | 集群作业收尾阶段可能报错；更稳的写法是 `if [ "${REMOVE_CODE_COPY_AFTER_JOB:-false}" = "true" ]; then`。另 `rm -rf $1` 对未加引号的路径同样有风险 |
| 8 | **集群脚本依赖两个未展开的环境文件**：开头无条件 `source $SCRIPT_DIR/.env.cluster` 与 `source $SCRIPT_DIR/../.env.base`，而 `docker/cluster/` 在清单中只展开 4 个 `.sh`，`docker/` 的 `.env.*` 也未展开 | `run_singularity.sh` + 文件树 | 直接运行会 `No such file or directory`，并且后续 `$CLUSTER_ISAAC_SIM_CACHE_DIR`、`$CLUSTER_SIF_PATH`、`$TMPDIR` 全为空 → `cp -r` / `tar -xf` / `singularity exec` 链式失败。使用前必须自建这两个 env 文件 |
| 9 | **Docker 以 root + host 网络运行**：`OMNI_KIT_ALLOW_ROOT=1`、`entrypoint: bash`、`stdin_open/tty: true`、`network_mode: host`，两个 service 均无端口隔离 | `docker/docker-compose.yaml` | 容器内 root 进程与宿主机共享网络命名空间，且 `../source`、`../scripts`、`../docs`、`../tools` 以可写 bind 挂入。仅建议在受控开发机使用，勿直接暴露到不可信网络 |
| 10 | **CloudXR 补丁一次开放 7 个端口并接受环境变量传入的 EULA**：`48010/tcp` + `47998/47999/48000/48005/48008/48012/udp`，`environment: - ACCEPT_EULA=${ACCEPT_EULA}` | `docker/docker-compose.cloudxr-runtime.patch.yaml` | 若 `ACCEPT_EULA` 已在 `.env` 中固化，等同于预先接受条款；部署前核对取值与防火墙策略 |
| 11 | **Windows 文档构建命令依赖 `make`**：`docs/README.md` 的 Windows `<details>` 块用 `:: 2. Build ... make current-docs`，批处理注释符 `::` 也说明作者按 Windows 写，但未提 `make` 来源 | `docs/README.md` 原文 | 原生 Windows 无 `make`，需 WSL / Git Bash / chocolatey make；文档未提示该前置条件 |
| 12 | **`.gitignore` 与 `docs/Makefile` 的目录约定一致但注释易误读**：`**/docs/_build/*` 忽略文档产物，`BUILDDIR = _build`；另有 `!.vscode` 试图放行顶层 `.vscode`，但 `**/.vscode/` 规则在前 | `.gitignore` + `docs/Makefile` | 顶层 `.vscode/` 在清单里确实有 2 个文件，说明放行有效；新增 IDE 配置时仍建议 `git check-ignore -v` 实测 |
| 13 | **数量断言无法复核**：上游 README 自述 `more than 16` 个机器人模型、`more than 30` 个环境，事实包清单未展开 `source/`，无法重数 | 现有 README Key Features + 文件树 | 本 README 一律标注为「上游自述」，不把这两个数字当作已核实结论 |
| 14 | **`tests/` 与 `source/` 等主体目录未展开**，因此本文未列任何具体 Python 包名 / 环境注册名 / 训练脚本路径 | 文件树节选 | 需要查具体环境或脚本时，请到上游仓库或本地 `source/`、`scripts/` 目录确认，不要按本 README 推断 |
| 15 | **许可组合复杂**：主体 BSD-3-Clause、`isaaclab_mimic` Apache-2.0，而 `docs/licenses/dependencies/` 下含 `isaacsim-license.txt`、`cuRobo-license.txt` 等**专有许可**组件；Isaac Sim 本体亦含专有许可条款 | `LICENSE` + 现有 README License 节 + `docs/licenses/` 清单 | 再分发或商用前须逐项核对；见[归属与许可证](#归属与许可证) |

---

## 归属与许可证

**本仓库是第三方副本，非原创；含大量上游第三方包与资产，权利归各自原作者 / 权利人所有。**

| 范围 | 许可 | 依据 |
|---|---|---|
| Isaac Lab 框架主体 | **BSD-3-Clause** | `LICENSE` 全文：`Copyright (c) 2022-2025, The Isaac Lab Project Developers`、`SPDX-License-Identifier: BSD-3-Clause`、三条再分发条件与免责声明 |
| `isaaclab_mimic` 扩展及其对应独立脚本 | **Apache-2.0** | 现有 README License 节 + `LICENSE-mimic`（GitHub 元数据确认根目录存在） |
| `source/isaaclab_mimic/`、`scripts/imitation_learning/isaaclab_mimic/` 的 `.py` | Apache-2.0 头 | `.pre-commit-config.yaml` 的第二个 `insert-license` 钩子 + `.github/LICENSE_HEADER_MIMIC.txt` |
| 其余 `.py` / `.yaml` | BSD-3-Clause 头 | `.pre-commit-config.yaml` 第一个 `insert-license` 钩子 + `.github/LICENSE_HEADER.txt` |
| 依赖与资产 | 各自原始许可 | [`docs/licenses`](docs/licenses)：`dependencies/` 下有 `isaacsim-license.txt`、`cuRobo-license.txt`、`legged-gym-license.txt`、`onnx-license.txt` 等；`assets/` 下有 `anymal_b`、`anymal_c`、`franka`、`kinova`、`robotiq`、`unitree` 六个许可文本 |
| Isaac Sim 本体 | **含专有许可组件**，不随本仓库分发 | 现有 README License 节：*Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms* |
| cuRobo | **专有许可条款** | 现有 README：`isaaclab_mimic` 需要 cuRobo，条款见 `docs/licenses/dependencies/cuRobo-license.txt` |

**原文要点**（`LICENSE`）：允许源码 / 二进制形式再分发与修改，条件是①源码再分发须保留版权声明、条件列表与免责声明；②二进制形式再分发须在文档 / 附带材料中复现这些内容；③未经事先书面许可，不得用版权方或其贡献者名称为衍生产品背书或推广。软件按 "AS IS" 提供，不承担任何直接或间接损害责任。

**本副本的附加声明**：新增的 `check_x30.py` 之外，作者自述其余均为上游原样；但 `docker/docker-compose.yaml` 存在本地离线资产挂载定制（见[已知问题](#已知问题)）。使用本副本做研究请按 `CITATION.cff` 引用上游技术报告，**不要以本副本名义主张原创贡献**。

---

## 引用

上游 README 原文要求：*If you use Isaac Lab in your research, please cite the technical report*。BibTeX 原样保留：

```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```

---

## 致谢

上游 README **Acknowledgement** 原文要点：Isaac Lab 的开发起始于 [Orbit](https://isaac-orbit.github.io/) 框架，并感谢 Orbit 作者们的基础性贡献。

`CONTRIBUTORS.md` 收录完整贡献者名单；`CONTRIBUTING.md` 内含 Developer Certificate of Origin 1.1 全文。

---

**本 README 依据事实包重写**：所有未在事实包「文件树」或「关键文件原文」中出现的具体值一律未写入；`source/`、`scripts/`、`tools/` 未展开，涉及具体包 / 环境 / 脚本路径请回源码确认。
