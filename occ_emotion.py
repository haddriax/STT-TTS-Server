from __future__ import annotations

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel

from config import OccConfig


class OccClassifyRequest(BaseModel):
    text: str


class OccClassifyResponse(BaseModel):
    emotion: str
    raw: str

logger = logging.getLogger(__name__)

OCC_LABELS = [
    "JOY", "DISTRESS", "HOPE", "FEAR", "SATISFACTION", "DISAPPOINTMENT",
    "RELIEF", "FEARS_CONFIRMED", "HAPPY_FOR", "PITY", "RESENTMENT", "GLOATING",
    "PRIDE", "SHAME", "ADMIRATION", "REPROACH", "LOVE", "HATE",
    "GRATITUDE", "ANGER", "GRATIFICATION", "REMORSE",
]
_LABELS_STR = ", ".join(OCC_LABELS)

_PROMPT = (
    "Tu es un classificateur d'émotions OCC. Analyse le texte et retourne UNIQUEMENT "
    "le label OCC le plus approprié parmi : {labels}.\n\n"
    "Texte : {text}\n"
    "Émotion OCC :"
)


def _extract_occ_label(text: str) -> str:
    upper = text.upper()
    for label in OCC_LABELS:
        if label in upper:
            return label
    return "JOY"


class OccEmotionClient:
    def __init__(self, cfg: OccConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._tokenizer = None

        if cfg.mode == "lora":
            self._load_lora()

    def _load_lora(self) -> None:
        import json

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = self.cfg.lora_model_path
        logger.info("Loading OCC LoRA model from %s...", path)

        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token

        with open(f"{path}/adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_name = adapter_cfg.get("base_model_name_or_path", "mistralai/Mistral-7B-Instruct-v0.2")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            logger.info("  OCC LoRA: loading with 4-bit quantization on CUDA")
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                quantization_config=bnb_config,
                device_map={"": 0},
                trust_remote_code=True,
            )
        else:
            logger.error("  OCC LoRA: CUDA unavailable — aborting")
            return
            logger.warning("  OCC LoRA: CUDA unavailable — loading in float32 on CPU (slow)")
            base = AutoModelForCausalLM.from_pretrained(
                base_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
            )

        self._model = PeftModel.from_pretrained(base, path)
        self._model.eval()
        logger.info("OCC LoRA model ready.")

    def _classify_lora(self, text: str) -> tuple[str, str]:
        import torch

        prompt = _PROMPT.format(labels=_LABELS_STR, text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        return _extract_occ_label(raw), raw

    async def _classify_ollama(self, text: str) -> tuple[str, str]:
        import httpx

        prompt = _PROMPT.format(labels=_LABELS_STR, text=text)
        payload = {
            "model": self.cfg.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 20},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.cfg.ollama_url}/api/generate", json=payload)
            resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return _extract_occ_label(raw), raw

    async def _run(self, text: str) -> tuple[str, str]:
        if self.cfg.mode == "lora":
            return await asyncio.to_thread(self._classify_lora, text)
        return await self._classify_ollama(text)

    async def classify(self, text: str) -> Optional[str]:
        """Classify text, returning only the OCC label. Returns None on failure."""
        try:
            label, _ = await self._run(text)
            return label
        except Exception as exc:
            logger.warning("OCC classification failed: %s", exc)
            return None

    async def classify_full(self, text: str) -> OccClassifyResponse:
        """Classify text, returning the OCC label and the raw LLM output."""
        label, raw = await self._run(text)
        return OccClassifyResponse(emotion=label, raw=raw)
