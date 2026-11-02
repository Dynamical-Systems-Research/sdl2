"""Native Tinker boundary for Pi, DeepSeek-V4.1-Flash-1017. v1.2 composition (2026-10-26): a copy of
deepseek-tinker-route-v22/ds_tinker_worker.py v3 (sha 74e2c919) with only MODEL, HF and TOKENIZER_HF changed; the base-model
check accepts the two 1017 ids. Original header (2026-10-10): Adapted from the Qwen route's tinker_worker.py
(qwen38-learning-execution-20261003/integration); rendering is usable offline, the client is lazy.

JSON-lines RPC, no HTTP proxy or scientific tools. The exact rendered ModelInput and sample token ids are saved so
scoring never reconstructs a privileged prefix. DeepSeek specifics follow the census rules that were checked token for
token against the HF chat template and the evaluation route's provider-reported prompt counts (58/58;
census-20261010/common.py):
  - tools: OpenAI-wrapped tools arrive from the transport; pass the function body with 'strict' removed (the cookbook
    wraps tools itself; 'strict' was never seen by the evaluation route's model);
  - user, tool and system content lists are joined with "\\n\\n" (the template's joined_text);
  - image parts are replaced by a text placeholder (this route has no image input; the count is returned per request).
Sampling: temperature 1.0, top_p 0.95 (no top_k; census settings), the renderer's stop sequences, max_tokens capped at
min(request, 32768, 262144 - prompt length) so a long context cannot overflow the window. A response whose DSML tool
calls do not parse (unparsed_tool_calls) is reported as parse_termination 'malformed' so the transport refuses to execute
it. The tokenizer is loaded from the local HF cache only (HF_HUB_OFFLINE; the route binds that cache read-only).
Version 2 (2026-10-10, after two transient 502s in p2): client creation and sampling are retried on transient service
errors (5xx, 429, timeouts) with waits of 5, 15, 45, 90 and 180 s; every retry is recorded in the receipt ("retries").
Version 3 (2026-10-11, after a service-wide Tinker outage from ~23:50Z in which the SDK raised "Failed to sample after
exhausting retries", which v2 did not match): that message is retried too, with waits of 10, 30, 60, 120, 240 and 480 s.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import tinker  # noqa: E402
from tinker_cookbook.renderers import ToolCall, get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144"
HF = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
# The 1017 tokenizer, config and encoding are byte-identical to the original repo's (HF git blobs 6a15814d, f3dad388,
# 09917a91; checked Oct 26). The offline cache holds only the original snapshot, so tokenization loads that id.
TOKENIZER_HF = "deepseek-ai/DeepSeek-V4.1-Flash"
RENDERER = "deepseekv4_1"
CTX = 262_144
PRICE = {"prefill": 0.60, "cached": 0.12, "sample": 3.60}          # $/M tokens (Tinker models page, Oct 10)
CREDENTIAL = Path("/home/jarrodbarnes/openreward-cookbook/training/tinker/.env")
RETRY_WAITS = (10, 30, 60, 120, 240, 480)   # seconds (~15 min total); v3 after the Oct 10-11 Tinker outage
TRANSIENT = ("502", "503", "504", "429", "bad gateway", "service unavailable", "gateway timeout", "timed out", "timeout",
             "connection reset", "connection aborted", "temporarily unavailable", "too many requests",
             "exhausting retries", "failed to sample", "unknown error")   # v3: the SDK's own give-up message


def with_retry(fn, retries):
    """Run fn(); on a transient service error wait and retry (bounded). Each retry is recorded in `retries`."""
    for i, wait in enumerate((0,) + RETRY_WAITS):
        if wait:
            time.sleep(wait)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - only transient service errors are retried; others re-raise at once
            msg = str(e).lower()
            if i == len(RETRY_WAITS) or not any(t in msg for t in TRANSIENT):
                raise
            retries.append({"attempt": i + 1, "error": str(e)[:200], "next_wait_s": RETRY_WAITS[i]})
IMAGE_PLACEHOLDER = "[image omitted: this model route has no image input]"


def load_credential():
    if not os.environ.get("TINKER_API_KEY") and CREDENTIAL.exists():
        for line in CREDENTIAL.read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().removeprefix("export ") == "TINKER_API_KEY":
                os.environ["TINKER_API_KEY"] = value.strip().strip("\"'")


def renderer_tools(tools):
    out = []
    for t in tools or []:
        f = t.get("function", t) if t.get("type") == "function" else t
        out.append({k: v for k, v in f.items() if k != "strict"})
    return out


def prepare(request):
    """Transport messages -> renderer messages (census rules); returns (messages, images_replaced)."""
    images = 0
    messages = []
    for src in request["messages"]:
        msg = dict(src)
        if isinstance(msg.get("content"), list):
            parts = []
            for part in msg["content"]:
                if part.get("type") == "image":
                    images += 1
                    parts.append({"type": "text", "text": IMAGE_PLACEHOLDER})
                elif part.get("type") in ("text", "thinking"):
                    parts.append(part)
                else:
                    raise ValueError(f"unsupported content part {part.get('type')!r}")
            msg["content"] = parts
            if msg["role"] in ("user", "tool", "system"):
                if any(p["type"] != "text" for p in parts):
                    raise ValueError("non-text part in a user/tool/system message")
                msg["content"] = "\n\n".join(p["text"] for p in parts)
        if msg.get("tool_calls"):
            msg["tool_calls"] = [ToolCall.model_validate(c) for c in msg["tool_calls"]]
        messages.append(msg)
    return messages, images


class Runtime:
    def __init__(self):
        self.tokenizer = get_tokenizer(TOKENIZER_HF)
        self.renderer = get_renderer(RENDERER, self.tokenizer)
        self.service = None
        self.clients = {}

    def client(self, checkpoint=None):
        if self.service is None:
            load_credential()
            self.service = tinker.ServiceClient()
        key = checkpoint or MODEL
        if key not in self.clients:
            self.clients[key] = self.service.create_sampling_client(
                **({"model_path": checkpoint} if checkpoint else {"base_model": MODEL}))
            actual = self.clients[key].get_base_model()
            if actual not in (MODEL, HF):
                raise ValueError(f"Unexpected base model: {actual}")
        return self.clients[key]

    def render(self, request):
        messages, images = prepare(request)
        if request.get("tools"):
            system = messages.pop(0)["content"] if messages and messages[0]["role"] == "system" else ""
            messages = self.renderer.create_conversation_prefix_with_tools(renderer_tools(request["tools"]), system) + messages
        return self.renderer.build_generation_prompt(messages), images

    def execute(self, request):
        started = time.monotonic()
        method = request.get("method", "sample")
        if method == "capabilities":
            self.client()
            return self.service.get_server_capabilities().model_dump(mode="json")
        if "model_input" in request:
            prompt, images = tinker.ModelInput.model_validate(request["model_input"]), 0
        else:
            prompt, images = self.render(request)
        model_input = prompt.model_dump(mode="json")
        prefix_hash = hashlib.sha256(json.dumps(model_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        receipt = {"model": MODEL, "checkpoint": request.get("checkpoint"), "renderer": RENDERER, "prompt_tokens": prompt.length,
                   "prompt_sha256": prefix_hash, "images_replaced": images, "model_input": model_input}
        if request.get("prefix_path"):
            Path(request["prefix_path"]).write_text(json.dumps(receipt) + "\n")
        if method == "render":
            return receipt
        retries = []
        client = with_retry(lambda: self.client(request.get("checkpoint")), retries)
        if method == "score":
            target = request["target_tokens"]
            combined = tinker.ModelInput(chunks=[*prompt.chunks, tinker.EncodedTextChunk(tokens=target)])
            response = with_retry(lambda: client.sample(combined, num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=1),
                                                        include_prompt_logprobs=True).result(), retries)
            values = response.prompt_logprobs[prompt.length:]
            if len(values) != len(target) or any(v is None for v in values):
                raise ValueError("Candidate likelihood alignment failed")
            receipt.update(target_tokens=target, logprobs=values, logprob_sum=sum(values), logprob_mean=sum(values) / len(values),
                           target_count=len(values), billed_prompt_tokens=combined.length,
                           generated_tokens=sum(len(s.tokens) for s in response.sequences), cached_tokens=response.prompt_cache_hit_tokens)
        else:
            room = CTX - prompt.length
            if room <= 0:
                raise ValueError(f"context overflow: prompt {prompt.length} tokens >= window {CTX}")
            params = tinker.SamplingParams(max_tokens=min(request.get("max_tokens", 32768), 32768, room),
                                           temperature=request.get("temperature", 1.0), top_p=0.95,
                                           stop=self.renderer.get_stop_sequences(), **({"seed": request["seed"]} if "seed" in request else {}))
            response = with_retry(lambda: client.sample(prompt, num_samples=request.get("num_samples", 1), sampling_params=params).result(),
                                  retries)
            sequences = []
            for seq in response.sequences:
                message, termination = self.renderer.parse_response(seq.tokens)
                content = message.get("content", [])
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                calls = [c.model_dump(mode="json") if hasattr(c, "model_dump") else c for c in message.get("tool_calls", [])]
                unparsed = message.get("unparsed_tool_calls") or []
                term = "malformed" if unparsed else str(termination)
                sequences.append({"tokens": seq.tokens, "logprobs": seq.logprobs, "sequence_id": seq.sequence_id,
                                  "stop_reason": seq.stop_reason, "parse_termination": term, "unparsed_tool_calls": len(unparsed),
                                  "message": {"role": "assistant", "content": content, "tool_calls": calls},
                                  "raw_text": self.tokenizer.decode(seq.tokens)})
            receipt.update(sequences=sequences, cached_tokens=response.prompt_cache_hit_tokens, max_tokens_used=params.max_tokens,
                           generated_tokens=sum(len(s.tokens) for s in response.sequences), billed_prompt_tokens=prompt.length)
        receipt["retries"] = retries
        receipt["wall_seconds"] = time.monotonic() - started
        n = request.get("num_samples", 1) if method == "sample" else 1
        p, c = receipt["billed_prompt_tokens"], receipt["cached_tokens"] or 0
        receipt["cost_estimate_usd"] = ((p - c) * PRICE["prefill"] + (c + (n - 1) * p) * PRICE["cached"]
                                        + receipt["generated_tokens"] * PRICE["sample"]) / 1e6
        receipt["billing_cost_status"] = "token_derived_estimate_not_invoice"
        return receipt


def main():
    runtime = Runtime()
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            result = runtime.execute(request)
            print(json.dumps({"id": request.get("id"), "result": result}), flush=True)
        except Exception as e:  # noqa: BLE001 - no credential or HTTP body in the RPC error channel
            print(json.dumps({"id": request.get("id"), "error": {"type": type(e).__name__, "message": str(e)[:500]}}), flush=True)


if __name__ == "__main__":
    main()
