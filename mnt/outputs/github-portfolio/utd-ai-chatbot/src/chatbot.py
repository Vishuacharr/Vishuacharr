"""
UTDChatbot — inference wrapper for the fine-tuned Llama 3.2 model.
Supports both local weights and HuggingFace Hub models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread


SYSTEM_PROMPT = """You are the UTD Academic Assistant, an expert on The University of Texas at Dallas.
You help students with course requirements, degree plans, academic policies, and campus resources.
Always be helpful, accurate, and concise. If you're unsure, say so."""


@dataclass
class Message:
    role: str    # "user" | "assistant" | "system"
    content: str


@dataclass
class ChatConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    load_in_4bit: bool = True


class UTDChatbot:
    """
    Fine-tuned Llama 3.2 chatbot for UTD academic queries.

    Usage:
        bot = UTDChatbot(model_path="./fine_tuned_model")
        print(bot.chat("What GPA do I need for the CS honors program?"))
    """

    def __init__(
        self,
        model_path: str = "meta-llama/Llama-3.2-3B-Instruct",
        cfg: Optional[ChatConfig] = None,
    ):
        self.cfg = cfg or ChatConfig()
        self.history: List[Message] = []
        self._load(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, stream: bool = False) -> str:
        """Single-turn or multi-turn chat."""
        self.history.append(Message(role="user", content=user_message))
        prompt = self._build_prompt()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        if stream:
            return self._stream_generate(inputs)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
                repetition_penalty=self.cfg.repetition_penalty,
                do_sample=self.cfg.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        self.history.append(Message(role="assistant", content=response))
        return response

    def stream_chat(self, user_message: str) -> Iterator[str]:
        """Yield tokens one-by-one for streaming UIs."""
        self.history.append(Message(role="user", content=user_message))
        prompt = self._build_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            do_sample=self.cfg.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        full_response = ""
        for token in streamer:
            full_response += token
            yield token

        self.history.append(Message(role="assistant", content=full_response.strip()))

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in self.history[-10:]:   # last 10 turns
            messages.append({"role": msg.role, "content": msg.content})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _load(self, model_path: str) -> None:
        print(f"[UTDChatbot] Loading model from: {model_path}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=self.cfg.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ) if self.cfg.load_in_4bit else None

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        print("[UTDChatbot] Model loaded ✅")
