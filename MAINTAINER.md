# 维护者手册

接手这个仓库的人看这一份就够了。目标：**日常只改参数和提示语，不碰架构**。

---

## 1. 一句话架构

```
Flask(app.py) ── 起子进程 ──▶ separate.py（分离/去混响/去和声）   ← py311 + torch cu121
                           ├─▶ rvc_convert.py（变声）            ← 另一个 python 3.9 + torch cu118
                           └─▶ mix.py（混音/混响/限幅）           ← py311
```

三个脚本都能独立命令行跑；`pipeline.py` 只是把三步串起来；网页界面只是给这三个脚本套了个壳。
**没有任何数据库、没有后台常驻任务、没有云服务。**

---

## 2. 文件地图（改东西先看这里）

| 你想改… | 改这个文件 | 注意 |
|---|---|---|
| 默认参数 / 预设 | `tools/webui/static/index.html`（滑杆 `value=`、预设按钮、`syncLabels()`） | 同时改 `app.py` 的 `rvc_cmd()` 兜底默认和 `pipeline.py` 的 argparse 默认，否则命令行与界面不一致 |
| 参数说明气泡 | `index.html` 里的 `data-tip="..."` | 只影响文案，不动指纹 |
| 参数指纹组成 | `app.py` 的 `tag_of()` | 加参数要写进 `payload`，否则不同参数生成同名文件 |
| 界面文案 / 布局 | `index.html` | 纯前端，刷新即生效（可能要硬刷新） |
| 分离流程（加一步处理） | `tools/separate.py` 的 `separate()` | 每加一步都要在 `app.py` 的 `/api/separate` 里加开关和 steps 标签 |
| RVC 调用方式 | `tools/rvc_convert.py` | 内核接口见第 4 节 |
| 混音规则 | `tools/mix.py` | 混响要**因果卷积**，别用 `mode="same"`（会让人声前出现预回声） |
| 外部路径 | `tools/settings.py` + `settings.json` | 代码里禁止写死盘符，一律走 settings |
| 常见报错 | `DEPLOY.md` 第 9 节 | 出现新坑就往这里加一行 |

---

## 3. 三种最常见的维护任务

### A. 只调参数（占 90%）

1. 改 `index.html` 里对应滑杆的 `value`、`#presetStable`/`#presetHard` 的复位值；
2. 改 `app.py` `rvc_cmd()` 里同名兜底默认；
3. 改 `pipeline.py` 里对应 `add_argument` 的默认值；
4. 改 `docs/PARAMETERS.md` 表格里的默认列；
5. `python tools\check_env.py` → 开界面，选一首歌，截 20 秒试听；
6. 提交 + push（见第 6 节）。

> 参数指纹会变，这是正常的。不要为了让旧文件名不变而去动 `tag_of()`。

### B. 加一条"失误提示"

改 `index.html` 的 `data-tip`，或 `app.py` 里作业完成的 `steps_label` 文案。
如果是要**自动拦截**某种坏输入，放在 `separate.py` 的兜底逻辑里（参考现成的去混响保护：
降幅 > 10 dB 就跳过并打日志）。

### C. 加一个分离/后处理步骤

1. `separate.py` 写一个 `xxx_vocals(path, outdir, ...) -> Path`，把结果写成规范文件名；
2. `separate()` 加参数、编排顺序、打印日志（日志格式 `xxx: ...`，`app.py` 靠它做步骤标签）；
3. `app.py` 的 `/api/separate` 加开关 + steps 标签；
4. `index.html` 第①步加一个 select；
5. 在 `DEPLOY.md` 第 5 节模型表里登记新模型和大小。

---

## 4. RVC 内核接口（唯一必须知道的契约）

```python
# rvc_core/infer/modules/vc/modules.py
vc = VC(config)                     # config 是单例，arg_parse() 会读 sys.argv → 必须先 sys.argv = sys.argv[:1]
vc.get_vc(model_basename)           # 读 os.environ["weight_root"] 下的 .pth
info, (sr, audio) = vc.vc_single(
    sid, input_audio_path, f0_up_key, f0_file, f0_method,
    file_index, file_index2, index_rate, filter_radius,
    resample_sr, rms_mix_rate, protect)
```

三个必须先设好的环境变量：`weight_root`、`index_root`、`rmvpe_root`，
并且**必须先 `os.chdir(rvc_core)`** 再 import（hubert 走相对路径 `assets/hubert/hubert_base.pt`）。

两个已知的坑（已在代码里绕过，别改回去）：

- 该内核返回的是 **int16 量级**数组，不是 [-1,1] 浮点 → 写盘前 `if abs(audio).max() > 1.5: audio /= 32768`；
- `faiss.read_index` 在 Windows 打不开**非 ASCII 路径**，会静默退化成"不用索引" → 代码把 index 复制到 `work/_indices/` 的 ASCII 路径再加载。

---

## 5. 改完怎么验证（5 分钟）

```bat
python tools\check_env.py                    :: 结构自检，全绿
python tools\separate.py "某首歌.mp3" -o work\smoke   :: 分离能跑（会真下模型）
:: 界面里：截 20 秒 → 生成对比 → 听
```

没有素材就随便找一首 30 秒以上的歌。**验证三件事**：能跑完、输出不削波（峰值 < 0.99）、参数指纹符合预期。

---

## 6. 提交与发布

```bat
git add -A
git commit -m "tune: <一句话说清改了什么>"
git push
```

- 只改文档/参数：`tune:` / `docs:` 开头；
- 改流程：`feat:`；修 bug：`fix:`；
- 发版打个 tag：`git tag v1.1 && git push --tags`（可选，README 不需要改版本号）。

推到 main 就等于发版——本项目没有 CI、没有构建步骤，克隆下来直接能用。

---

## 7. 不要做的事

| 别做 | 原因 |
|---|---|
| 把 `settings.json` / 模型 / `work/` 提交进仓库 | 体积巨大且含个人路径，`.gitignore` 已挡 |
| 在代码里写死 `D:\`、`C:\Users\...` | 换电脑就废，一律走 `settings.py` |
| 用 `fftconvolve(mode="same")` 做混响 | 湿信号会前移半个 IR，人声前面出现预回声 |
| 用 f0 跳变率给音质打分 | 与听感无关，会误导参数决策（历史上误导过很多次） |
| 为了"更像"而默认开去和声 | 会削掉和声/叠唱，多数歌因此变薄、口胡反而增加 |
| 改 `tag_of()` 的 key 顺序/类型 | 老文件的指纹全部失效，无法追溯"这版我听过" |

---

## 8. 已知的未完成项

- 去和声只有"开/关"，没有强度；靠换更温和的 karaoke 模型代替。
- `filter_radius` 对 crepe/rmvpe 无效（内核限制），界面已注明。
- 没有自动化测试；回归靠 `check_env.py` + 20 秒试听。
- 实时变声参数（formant / block_time / 降噪）在离线内核里不存在，只在界面上做灰色对照。