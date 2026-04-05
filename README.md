# STT-TTS-Server

A local FastAPI server combining speech transcription, text-to-speech with real-time ARKit facial animation, and OCC emotion classification. All models run in-process — no external services required beyond optional Ollama for emotion classification.

- **STT** — Whisper via `faster-whisper`, output as JSON, SRT, VTT, or plain text
- **TTS + ARKit** — Kokoro synthesis + IPA→viseme→ARKit blendshape pipeline at configurable FPS
- **OCC Emotion** — 22-label OCC classifier appended to STT responses (Ollama or fine-tuned LoRA)

---

## Requirements

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/) package manager
- NVIDIA GPU with CUDA 12.x recommended (CPU-only works, STT is slower)

---

## Installation

```bash
git clone <repo>
uv sync
```

Models are downloaded from HuggingFace on first run.

**CPU-only setup** — set in `config.yaml`:
```yaml
model:
  device: cpu
  compute_type: int8
```

---

## Running

```bash
uv run start.py
```

The server starts on `http://127.0.0.1:8000` by default. Interactive API docs are at `/docs`.

Startup order: Whisper loads → Kokoro loads → OCC client instantiates (if enabled) → health checks run.

---

## Configuration

Edit `config.yaml`. The top-level `language` key (`fr`, `en`, `en-gb`) auto-derives the Whisper language, Kokoro voice, and IPA mapping — override per-component only if needed.

**Key settings**

| Key | Default | Description |
|-----|---------|-------------|
| `language` | `fr` | BCP-47 code: `fr` \| `en` \| `en-gb` |
| `model.name` | `turbo` | Whisper model: `tiny` `base` `small` `medium` `large-v3` `turbo` |
| `model.device` | `auto` | `cuda` \| `cpu` \| `auto` |
| `kokoro.default_voice` | derived | e.g. `af_bella`, `bf_emma`, `ff_siwis` |
| `kokoro.fps` | `60` | ARKit frame rate (`24`, `30`, `60`) |
| `kokoro.arkit_level` | `2` | Blendshape detail: `1` basic · `2` advanced · `3` full suit |
| `kokoro.output_sample_rate` | `48000` | WAV output Hz (Kokoro native is 24000) |
| `occ.enabled` | `false` | Append emotion label to STT responses |
| `occ.mode` | `ollama` | `ollama` \| `lora` |
| `occ.max_vram_gb` | `null` | Cap OCC VRAM; excess spills to RAM (`lora` mode only) |

Environment variable overrides: `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE`, `WHISPER_HOST`, `WHISPER_PORT`, `WHISPER_LOG_LEVEL`, `WHISPER_CONFIG`.

---

## API

See [`ROUTES.md`](ROUTES.md) for full documentation with request fields, response schemas, and curl examples.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + loaded model name |
| `GET` | `/models` | Loaded Whisper model |
| `POST` | `/stt/transcribe` | Transcribe audio → text (JSON, SRT, VTT, plain) |
| `POST` | `/stt/translate` | Transcribe + translate to English |
| `POST` | `/tts/arkit` | Synthesise speech + ARKit blendshape frames |
| `POST` | `/tts/words` | Synthesise speech + word timestamps _(deprecated)_ |
| `POST` | `/emotion/classify` | Classify text → OCC emotion label _(if `occ.enabled`)_ |

Quick example:

```bash
# Transcribe
curl -X POST http://localhost:8000/stt/transcribe -F file=@audio.mp3

# Synthesise with blendshapes, decode audio
curl -s -X POST http://localhost:8000/tts/arkit \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' \
  | jq -r '.audio' | base64 -d > out.wav
```

---

## Language support

| `language` key | Whisper | Kokoro code | Default voice |
|----------------|---------|-------------|---------------|
| `fr` | `fr` | `f` | `ff_siwis` |
| `en` | `en` | `a` | `am_adam` |
| `en-gb` | `en` | `b` | `bf_emma` |

The IPA→viseme pipeline automatically selects the matching mapping file (`tts/ipa_to_viseme_fr.json` for French, `tts/ipa_to_viseme.json` for English variants).

---

## OCC Emotion

When `occ.enabled: true`, `POST /stt/transcribe` and `POST /stt/translate` append an `"emotion"` field to `json` and `verbose_json` responses. The classifier picks one of 22 OCC labels:

`JOY` `DISTRESS` `HOPE` `FEAR` `SATISFACTION` `DISAPPOINTMENT` `RELIEF` `FEARS_CONFIRMED` `HAPPY_FOR` `PITY` `RESENTMENT` `GLOATING` `PRIDE` `SHAME` `ADMIRATION` `REPROACH` `LOVE` `HATE` `GRATITUDE` `ANGER` `GRATIFICATION` `REMORSE`

Two backends:
- **`ollama`** — requires [Ollama](https://ollama.com) running locally with the configured model pulled (`ollama pull mistral`)
- **`lora`** — loads a fine-tuned Mistral-7B LoRA adapter in-process with 4-bit quantization (CUDA required); adapter path set via `occ.lora_model_path`

Classification failures are non-fatal: the `emotion` field is silently omitted.
