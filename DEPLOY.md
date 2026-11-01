# 全新电脑从零部署

目标：一台干净的 Windows 上，跑起和作者相同的环境与参数。
全程只写文件、不装系统级东西（除了 ffmpeg 和一个 Python）。

预计耗时：**40~70 分钟**（其中 20 分钟是下载依赖和模型）。

---

## 0. 先认清要准备好三样东西

| 需要的东西 | 用途 | 从哪来 |
|---|---|---|
| Python **3.11** | 跑分离 / 去混响 / 混音 / 网页服务 | python.org |
| Python **3.9 + torch 2.0.0+cu118** | 跑 RVC 变声 | 任意 RVC 打包版自带的 `runtime/`，或自己建 conda 环境 |
| ffmpeg | 解码 mp3、转码成品 | gyan.dev 的 win64 解压版 |

> 第二个 Python 不用单独装：绝大多数 RVC 整合包（RVCMAX / Applio / RVC-WebUI）里都带一个
> `runtime\python.exe`，直接拿它用最省事，本项目只读它、不改它。

---

## 1. 装 Python 3.11

下载 <https://www.python.org/downloads/release/python-3119/> 的 **Windows installer (64-bit)**，
安装时勾上 `Add python.exe to PATH`。

验证：

```bat
python -V
```

便携版（不想动系统）也可以：下 `python-3.11.9-embed-amd64.zip` 解压到 `py311\`，
把 `py311\python311._pth` 改成

```
python311.zip
.
Lib\site-packages
import site
```

然后 `py311\python.exe get-pip.py`（get-pip 从 <https://bootstrap.pypa.io/get-pip.py> 下载）。

---

## 2. 装依赖

先装 torch（必须带 CUDA 版本，别用默认 CPU 版）：

```bat
python -m pip install torch==2.4.1+cu121 torchaudio==2.4.1+cu121 ^
  --index-url https://download.pytorch.org/whl/cu121
```

再装其余：

```bat
python -m pip install -r requirements.txt
```

验证 GPU 可用：

```bat
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

期望输出 `2.4.1+cu121 True`。**`False` 就先别往下走**，说明装成 CPU 版了。

国内网络慢可以加 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

---

## 3. 装 ffmpeg

下 <https://www.gyan.dev/ffmpeg/builds/> 的 `ffmpeg-release-full.7z`，解压到比如
`C:\tools\ffmpeg\`，把 `bin` 加进 PATH，或者直接在 `settings.json` 里写全路径。

验证：`ffmpeg -version`。

---

## 4. 准备 RVC 内核（关键一步，决定能不能变声）

本项目的思路：**不碰你原来的 RVC 安装，只把它的内核复制一份出来单独跑**。

在仓库根目录建 `rvc_core\`，从 RVC 整合包里复制这几样（都是明文 .py）：

```
rvc_core\
├── infer\            ← 整个目录（modules/vc、lib、uvr5 …）
├── configs\          ← 整个目录（config.py、inuse\ 里的 json）
└── assets\
    ├── hubert\hubert_base.pt
    ├── rmvpe\rmvpe.pt
    └── pretrained_v2\{G,D,f0G,f0D}{32k,40k,48k}.pth
```

再确认 `rvc_core\infer\modules\vc\modules.py` 里存在：

```python
class VC:
    def vc_single(self, sid, input_audio_path, f0_up_key, f0_file, f0_method,
                  file_index, file_index2, index_rate, filter_radius,
                  resample_sr, rms_mix_rate, protect):
```

以及 `rvc_core\infer\modules\vc\pipeline.py` 里有 `index.search(npy, k=8)` 这一行
（本项目把它改成了可调的 `INDEX_K`，没有这行说明内核版本不对）。

> 没有 RVC 整合包？装官方 WebUI 也行：<https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>，
> 用它的 `runtime\python.exe` + 复制 `infer/`、`configs/`、`assets/`。

---

## 5. 放模型

**分离模型** → `models\mdxc\`。首次运行时 audio-separator 会自己下载；也可以手动放到该目录：

| 文件 | 作用 | 大小 |
|---|---|---|
| `mel_band_roformer_kim_ft2_bleedless_unwa.ckpt` | **默认** 人声/伴奏分离 | ~913 MB |
| `mel_band_roformer_karaoke_becruily.ckpt` | 去和声（主唱 vs 其余） | ~1.7 GB |
| `dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt` | 去混响 / 去回声 | ~456 MB |

下载源（audio-separator 的公共模型仓库）：

```
https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/<文件名>
```

**RVC 模型** → `rvc_models\<任意名字>\`，每个文件夹放 `.pth`（必须）+ `.index`（强烈建议）：

```
rvc_models\
└── my-model\
    ├── my-model.pth
    └── my-model.index
```

界面第②步可以直接「⬆ 导入模型」，也可以手动复制。

---

## 6. 配置 settings.json

复制 `settings.example.json` → `settings.json`，只改这几个值：

```json
{
  "rvc_python": "E:/RVC/runtime/python.exe",
  "ffmpeg": "C:/tools/ffmpeg/bin/ffmpeg.exe",
  "rvc_core": "D:/AI/CoverStudio/rvc_core",
  "sep_model_dir": "D:/AI/CoverStudio/models/mdxc",
  "model_roots": [
    {"path": "D:/AI/CoverStudio/rvc_models", "label": "本地模型", "writable": true},
    {"path": "E:/RVC/User_Data/models", "label": "RVC runtime（只读）", "writable": false}
  ]
}
```

- 路径用 `/` 或 `\\` 都行；`settings.json` 不在 git 里，不会被提交。
- 换电脑只改这一个文件，所有工具都读它。

---

## 7. 启动

双击 `start.bat`（后台 `pythonw` 启动，关掉黑窗口服务不会死），浏览器打开
<http://127.0.0.1:8765>。

停止：`stop.bat`，或界面右上角「⏻ 退出」。

---

## 8. 命令行（不走界面）

```bat
:: 只分离
python tools\separate.py "song.mp3" -o work\out --strip-backing --dereverb

:: 只变声（用第 4 步的 python 3.9）
"%RVC_PYTHON%" tools\rvc_convert.py --input work\out\vocals.wav --output work\out\conv.wav ^
  --model-dir rvc_models\my-model --model my-model.pth --index my-model.index ^
  --index-rate 0.50 --protect 0.40 --rms-mix-rate 0.30 --filter-radius 7 ^
  --pre-gain 10 --f0-method crepe --max-segment 100 --index-k 28

:: 只混音
python tools\mix.py --vocals work\out\conv.wav --instrumental work\out\instrumental.wav ^
  -o work\final\song.mp3 --vocal-gain 0 --instrumental-gain 0 --format mp3

:: 三步一次跑完
python tools\pipeline.py "song.mp3" --model-dir rvc_models\my-model --model my-model.pth -o work\final
```

---

## 9. 常见错误

| 现象 | 原因 / 解法 |
|---|---|
| `CUDA is available in Torch` 没出现 | torch 装成 CPU 版，重装 cu121 |
| `SSLError: CERTIFICATE_VERIFY_FAILED` | 公司/校园网的 HTTPS 拦截代理。跑一次 `powershell -ExecutionPolicy Bypass -File tools\export_roots.ps1`，把生成的 `tmp\win_roots.pem` 和 certifi 的 `cacert.pem` 拼起来，设 `SSL_CERT_FILE` 指向它 |
| `faiss ... could not open ... for reading` | 模型路径含中文。代码已自动把 index 复制到 ASCII 缓存目录；如果还报，直接把模型放到全英文路径 |
| `ffmpeg error / Illegal byte sequence` | 输入文件名含中文/日文且经由命令行传参；改成先用 ASCII 名复制一份 |
| RVC 报 `ModuleNotFoundError: ffmpeg` | 用了错误的 python 跑 `rvc_convert.py`，必须用第 4 步那个 3.9 runtime |
| `TypeError: 'float' object cannot be interpreted as an integer` | 改了 `--max-segment` 相关的 x_center/x_query，必须是 int |
| 网页显示「未找到 RVC Runtime」 | `settings.json` 里 `rvc_python` 路径不对 |
| 端口被占用 / 双击没反应 | `stop.bat` 先清一遍；服务是后台进程，改代码后要重启 |
| 输出削波（波形全红） | 某个内核返回的是 int16 量级数组（见 `rvc_convert.py` 里 `if abs(audio).max() > 1.5: audio /= 32768`），换内核版本或确认这段还在 |
| 去混响后人声几乎没了 | 该素材混响太重，模型把主体也当混响剥了。默认已有兜底（降幅 >10 dB 自动跳过），可手动关掉去混响 |

---

## 10. 性能参考（RTX 4050 Laptop 6 GB）

| 操作 | 5 分半的整首歌 |
|---|---|
| 分离（一趟） | 约 3 分 20 秒 |
| 去和声（第二趟） | 约 3 分 30 秒 |
| 去混响 | 约 0.35~0.6 倍时长 |
| RVC 变声（单段 100 s / index_k 28） | 约 0.25 倍时长 |
| 混音 | 2~4 秒 |

显存吃紧就调小「单段推理时长」（65 s 是内核原值，最省显存也最块；调大更连贯但耗时按平方增长）。