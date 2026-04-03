# API Routes

**Base URL:** `http://127.0.0.1:8000` (configure `server.host` / `server.port` in `config.yaml`)

**Interactive docs (Swagger UI):** `http://127.0.0.1:8000/docs`

---

## GET /health

Returns the server status and the name of the loaded Whisper model.

**Response `200`**
```json
{
  "status": "ok",
  "model": "small"
}
```

```bash
curl http://localhost:8000/health
```

---

## GET /models

Returns the currently loaded Whisper model. Useful for programmatic checks.

**Response `200`**
```json
{
  "data": [
    {"id": "small", "object": "model"}
  ]
}
```

```bash
curl http://localhost:8000/models
```

---

## POST /stt/transcribe

Transcribe an audio file to text using Whisper. Accepts any format supported by ffmpeg (mp3, wav, ogg, flac, m4a, webm, …).

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | **yes** | — | Audio file to transcribe |
| `language` | string | no | auto-detect | Source language as a BCP-47 code (`en`, `fr`, `ja`, …). Omit to auto-detect. |
| `prompt` | string | no | — | Text hint to guide the model (e.g. a list of proper nouns, or the previous sentence) |
| `response_format` | string | no | `json` | Output format — see below |
| `temperature` | float | no | `0.0` | Sampling temperature. `0.0` = greedy (most deterministic). Increase slightly (e.g. `0.2`) if output loops or repeats. |

**`response_format` options**

| Value | Content-Type | Shape |
|-------|-------------|-------|
| `json` | `application/json` | `{"text": "…"}` |
| `verbose_json` | `application/json` | Full object with segments, timestamps, language, duration — see schema below |
| `text` | `text/plain` | Raw transcription string |
| `srt` | `text/plain` | SRT subtitle file |
| `vtt` | `text/vtt` | WebVTT subtitle file |

**OCC Emotion field** — when `occ.enabled: true` in `config.yaml`, `json` and `verbose_json` responses include an additional `"emotion"` field with one of the 22 OCC labels (`JOY`, `FEAR`, `ADMIRATION`, etc.). The field is omitted when OCC is disabled or classification fails.

**`verbose_json` response schema**
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 5.12,
  "text": "Hello, world. This is a test.",
  "emotion": "JOY",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.4,
      "text": " Hello, world.",
      "tokens": [50364, 2425, 11, 1002, 13],
      "temperature": 0.0,
      "avg_logprob": -0.18,
      "compression_ratio": 0.91,
      "no_speech_prob": 0.005
    },
    {
      "id": 1,
      "start": 2.4,
      "end": 5.12,
      "text": " This is a test.",
      "tokens": [50464, 639, 307, 257, 1500, 13],
      "temperature": 0.0,
      "avg_logprob": -0.12,
      "compression_ratio": 0.88,
      "no_speech_prob": 0.002
    }
  ]
}
```

**Error responses**

| Status | Cause |
|--------|-------|
| `400` | Invalid `response_format` value |
| `422` | Audio could not be decoded or transcribed |

**Examples**

```bash
# Plain JSON (default)
curl -X POST http://localhost:8000/stt/transcribe \
  -F file=@recording.mp3

# Specify language, get verbose output
curl -X POST http://localhost:8000/stt/transcribe \
  -F file=@recording.wav \
  -F language=fr \
  -F response_format=verbose_json

# SRT subtitles
curl -X POST http://localhost:8000/stt/transcribe \
  -F file=@recording.mp3 \
  -F response_format=srt

# With a prompt hint
curl -X POST http://localhost:8000/stt/transcribe \
  -F file=@recording.mp3 \
  -F prompt="Speaker names: Alice, Bob."
```

---

## POST /stt/translate

Transcribe audio and translate the result to **English**. Accepts the same audio formats as `/stt/transcribe`. Source language is always auto-detected.

**Request** — `multipart/form-data`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | **yes** | — | Audio file to translate |
| `prompt` | string | no | — | Text hint to guide the model |
| `response_format` | string | no | `json` | Same options as `/stt/transcribe` |
| `temperature` | float | no | `0.0` | Sampling temperature |

**Response** — identical shape to `/stt/transcribe` for the chosen `response_format`, always in English. Includes `"emotion"` when `occ.enabled: true` and format is `json` or `verbose_json`.

```bash
# Translate a French recording to English
curl -X POST http://localhost:8000/stt/translate \
  -F file=@french_audio.mp3

# Get subtitles in English
curl -X POST http://localhost:8000/stt/translate \
  -F file=@japanese_audio.mp3 \
  -F response_format=vtt
```

---

## TTS endpoints

All three routes share the same response schema. Fields absent when not applicable.

Each route is enabled independently via boolean flags in `config.yaml → kokoro`:

| Config flag | Route | Default | Status |
|------------|-------|---------|--------|
| `activate_base_arkit: true` | `POST /tts/arkit` | enabled | stable |
| `activate_words: true` | `POST /tts/words` | enabled | **deprecated** |

Disabled routes return `404`.

### Unified response schema

```json
{
  "audio": "<base64-encoded wav>",
  "format": "wav",
  "duration": 1.85,
  "frames": [...],       // present on /tts/arkit
  "timestamps": [...]    // present on /tts/words (deprecated)
}
```

`frames` and `timestamps` are omitted (not `null`) when not applicable.

---

## POST /tts/arkit

Synthesise speech from text and return ARKit blend shape frames for driving a 3D avatar.

**Request** — `application/json`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | **yes** | — | Text to speak |
| `voice` | string | no | config default | Kokoro voice ID (e.g. `af_bella`, `af_sky`, `bf_emma`) |
| `speed` | float | no | config default (`1.0`) | Speech rate multiplier — `0.5` (slow) to `2.0` (fast) |

**Response `200`**

```json
{
  "audio": "<base64-encoded wav>",
  "format": "wav",
  "duration": 1.85,
  "frames": [
    {
      "time": 0.0,
      "blendshapes": {}
    },
    {
      "time": 0.016667,
      "blendshapes": {
        "jawOpen": 0.25,
        "mouthSmileLeft": 0.5,
        "mouthSmileRight": 0.5
      }
    }
  ]
}
```

**Frame notes:**
- `time` — seconds from the start of the audio clip
- `blendshapes` — only keys with non-zero weight are included; all other ARKit blend shapes are implicitly `0.0`
- FPS is controlled by `config.yaml → kokoro.fps` (typical values: `24`, `30`, `60`)
- Transitions between phonemes use a 50 ms cosine crossfade for smooth animation

**Examples**

```bash
curl -X POST http://localhost:8000/tts/arkit \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'

# Custom voice and speed
curl -X POST http://localhost:8000/tts/arkit \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "voice": "af_bella", "speed": 0.85}'

# Decode and save audio
curl -s -X POST http://localhost:8000/tts/arkit \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}' \
  | jq -r '.audio' | base64 -d > output.wav
```

---

## POST /tts/words ⚠️ Deprecated

> **Deprecated.** Use `/tts/arkit` instead. This route may be removed in a future release.

Synthesise speech from text and return word-level timestamps only (no blend shapes).

**Request** — same fields as `/tts/arkit`

**Response `200`**

```json
{
  "audio": "<base64-encoded wav>",
  "format": "wav",
  "timestamps": [
    {"word": "Hello",  "start_time": 0.0,  "end_time": 0.42},
    {"word": "world",  "start_time": 0.50, "end_time": 0.98}
  ]
}
```

**Examples**

```bash
curl -X POST http://localhost:8000/tts/words \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}'
```

