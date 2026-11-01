# AI Cover Studio · 本地 AI 翻唱流水线

把一首成品歌变成「你指定的音色」唱的版本，全程跑在自己电脑上：**不联网、不上传、不花钱（0 token）**。

```
歌曲 ──[1] 分离──▶ 人声 / 伴奏 ──[2] RVC 变声──▶ 新的人声 ──[3] 混音──▶ 成品
```

三个步骤在网页界面里各占一个面板，**互相解耦**：每一步都能自己传文件、自己导出，做完自动弹窗告诉你文件存在哪。

![流程图见 DEPLOY.md](docs/flow.txt)

## 它做了什么

| 阶段 | 用什么 | 说明 |
|---|---|---|
| ① 分离 | [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) + BS-Roformer / Mel-Band RoFormer | 44.1k 立体声，输出 `vocals.wav` / `instrumental.wav`；可选**去和声**、**去混响**、**伴奏回填原唱混响** |
| ② 变声 | RVC（任意 RVC 打包版的 `infer/` 内核） | 隔离运行，只读你的原安装；12 个内核参数 + 2 个额外调参口（单段推理时长、检索邻居数） |
| ③ 混音 | numpy / librosa / ffmpeg | 增益、软限幅、人声混响、wav/flac/mp3/m4a |

> 分离用 BS-Roformer 这类 SDR 12.9/17.0 的模型（比 MDXNet / UVR-MDX 高 1.5~3 dB），速度约 1.5 倍实时。

## 两套预设（界面第②步右上角，直接点）

| 参数 | **默认 · 硬骨头 B（推荐）** | 预设 A · 抒情/气声 |
|---|---|---|
| 变调 pitch | 0 | 0 |
| 检索比例 index_rate | 0.50 | 0.60 |
| 检索邻居 index_k | 28 | 10 |
| 响度混合 rms_mix_rate | 0.30 | 0.20 |
| 保护清辅音 protect | 0.40 | 0.47 |
| 滤波半径 filter_radius | 7 | 7 |
| 输入增益 pre_gain | +10 dB | +18 dB |
| 音高算法 f0 | crepe | rmvpe |
| 单段推理时长 | 100 s | 100 s |

- **B 是通用默认**，在多种音源 + 多个模型上复测过，咬字最稳。
- **A 只在特定场景更好**：想要更像模型本色（index 0.60）、气声/耳语多（protect 0.47）、原唱动态大（rms 0.20）、录音电平偏低（+18 dB 才推得动）。
- 第①步默认：去混响 **开**（人声被吃太多会自动跳过）、伴奏回填混响 **关**、去和声 **关**。

## 三步之外的关键经验

1. **「重复字 / 口胡」多半不是分离不干净，而是和声被一起喂进了 RVC**。歌里有明显叠唱时，第①步开「去和声」；开完人声变薄可把 pre_gain 拉高补偿。
2. **pre_gain 常被峰值限幅**：源干声峰值接近满量程时，+18 dB 实际只推得动 2~3 dB —— 这是物理限制，不是 bug。
3. **混响不能当「效果更好」的万能药**：先把干声修干净，再在第③步给人声加 10~20% 混响，比一开始就带混响去转换好听得多。
4. **f0 指标（跳音率、八度错误）和听感基本无关**，别用它判断音质，用耳朵。
5. `filter_radius` 这版内核**只对 harvest 生效**（`pipeline.py` 里 `medfilt` 只在 harvest 分支），crepe/rmvpe 下改了没效果。

## 快速开始

详见 [DEPLOY.md](DEPLOY.md)。三步：

```bat
:: 1. 装 Python 依赖（见 DEPLOY.md）
:: 2. 编辑 settings.json 指向你的 RVC runtime / ffmpeg / 模型目录
:: 3. 双击
start.bat
```

浏览器打开 <http://127.0.0.1:8765>。

## 目录

```
.
├── start.bat / stop.bat        启动、停止（后台 pythonw，关掉窗口不会杀服务）
├── settings.example.json       复制成 settings.json 后改路径，代码不用动
├── requirements.txt
├── tools/
│   ├── settings.py             所有外部路径的唯一来源
│   ├── separate.py             ① 分离（含去和声 / 去混响 / 回填混响）
│   ├── rvc_convert.py          ② RVC 变声（隔离内核副本）
│   ├── mix.py                  ③ 混音
│   ├── pipeline.py             命令行一次性跑完三步
│   ├── export_roots.ps1        （可选）导出系统根证书，绕过公司代理的 TLS 拦截
│   └── webui/app.py + static/  网页界面（Flask）
├── models/mdxc/                放分离模型（.ckpt，首次使用会自动下载）
├── rvc_core/                   放 RVC 内核副本（infer/ configs/ assets/）
├── rvc_models/<名字>/          放你自己的 .pth + .index
└── work/                       所有中间产物与成品（可随时删）
```

## 环境要求

- Windows 10/11 + NVIDIA 显卡（6 GB 显存可跑，建议 8 GB+）
- Python 3.11（跑分离/混音）+ **另一个 Python 3.9 + torch 2.0.0+cu118**（跑 RVC，通常由 RVC 打包版自带）
- ffmpeg 在 PATH 里
- 磁盘：依赖约 5 GB，模型约 2.5 GB

## 已知限制

- 只支持离线渲染；RVC 的实时参数（formant、block_time、降噪等）在离线内核里不存在，界面里只做灰色说明。
- 分离模型没有「强度」旋钮，效果强弱只能靠换模型。
- 去和声用的是 karaoke 模型，本质是「主唱 vs 其余」的二分类，和声很密的歌会削得偏狠。

## License

MIT © pinn