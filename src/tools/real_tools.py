"""Real tool execution backends for GTA / m&m verification runs.

Replaces the step-by-step mock when config.tool_execution == "real".
Backends (documented deviations from the benchmarks' official stacks):
  GTA  : OCR=easyocr, ImageDescription=BLIP-base, CountGivenObject=OWLv2,
         Calculator/Solver=sympy, GoogleSearch=ddgs (DuckDuckGo, no key)
  m&m  : text classification=distilbert-sst2; summarization / generation /
         question answering are backed by the local serving LLM (Qwen3-8B),
         an explicit substitution for m&m's per-tool checkpoints; facts=
         numbersapi.com; wikipedia simple search=MediaWiki REST API.

Models load lazily on first use and are pinned to `device`. Exceptions
propagate to ToolManager.execute, which attaches the typed error token.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request


class RealToolExecutor:
    def __init__(self, data_root: str = "", llm_client=None, device: str = "cuda:1"):
        self.data_root = data_root
        self.llm = llm_client
        self.device = device
        self._ocr = None
        self._captioner = None
        self._detector = None
        self._sentiment = None

    # ------------------------------------------------------------- dispatch

    def execute(self, tool_name: str, args: dict) -> object:
        handler = self._HANDLERS.get(tool_name)
        if handler is None:
            raise ValueError(f"No real backend for tool: {tool_name}")
        return handler(self, args or {})

    # ------------------------------------------------------------- helpers

    def _image_path(self, args: dict) -> str:
        rel = args.get("image") or args.get("path") or ""
        path = rel if os.path.isabs(rel) else os.path.join(self.data_root, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f"image not found: {rel}")
        return path

    def _llm_text(self, system: str, user: str) -> str:
        if self.llm is None:
            raise RuntimeError("LLM-backed tool requires an llm_client")
        return self.llm.generate(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}]
        ).strip()

    @staticmethod
    def _http_get(url: str, timeout: int = 15) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "tooltree-eval"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")

    # ------------------------------------------------------------- GTA tools

    def _ocr_tool(self, args):
        if self._ocr is None:
            import easyocr

            gpu = self.device if self.device.startswith("cuda") else False
            self._ocr = easyocr.Reader(["en"], gpu=gpu)
        results = self._ocr.readtext(self._image_path(args))
        lines = []
        for bbox, text, conf in results:
            x1 = int(min(p[0] for p in bbox)); y1 = int(min(p[1] for p in bbox))
            x2 = int(max(p[0] for p in bbox)); y2 = int(max(p[1] for p in bbox))
            lines.append(f"({x1}, {y1}, {x2}, {y2}) {text}")
        return "\n".join(lines) if lines else "(no text detected)"

    def _image_description(self, args):
        if self._captioner is None:
            from transformers import pipeline

            self._captioner = pipeline(
                "image-to-text", model="Salesforce/blip-image-captioning-base",
                device=self.device,
            )
        out = self._captioner(self._image_path(args), max_new_tokens=60)
        return out[0]["generated_text"].strip()

    def _count_given_object(self, args):
        if self._detector is None:
            from transformers import pipeline

            self._detector = pipeline(
                "zero-shot-object-detection",
                model="google/owlv2-base-patch16-ensemble", device=self.device,
            )
        label = args.get("text") or args.get("object") or ""
        dets = self._detector(
            self._image_path(args), candidate_labels=[label], threshold=0.3
        )
        return len(dets)

    def _calculator(self, args):
        import sympy

        expr = str(args.get("expression") or args.get("expr") or "")
        val = float(sympy.sympify(expr, evaluate=True).evalf())
        # render integers without trailing .0
        return str(int(val)) if val.is_integer() else str(val)

    def _solver(self, args):
        import sympy

        formula = str(args.get("formula") or args.get("equation")
                      or args.get("expression") or "")
        solutions = sympy.solve(formula)
        return str(solutions)

    def _google_search(self, args):
        from ddgs import DDGS

        query = str(args.get("query") or args.get("text") or "")
        k = int(args.get("k") or 5)
        rows = list(DDGS().text(query, max_results=k))
        if not rows:
            return "(no results)"
        return "\n".join(
            f"{i+1}. {r.get('title', '')} — {r.get('body', '')[:300]}"
            for i, r in enumerate(rows)
        )

    # ------------------------------------------------------------- m&m tools

    def _text_classification(self, args):
        if self._sentiment is None:
            from transformers import pipeline

            self._sentiment = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=self.device,
            )
        out = self._sentiment(str(args.get("text") or ""))[0]
        return f"{out['label']} (score={out['score']:.3f})"

    def _text_summarization(self, args):
        return self._llm_text(
            "You are a summarization tool. Return only the summary, 1-3 sentences.",
            f"Summarize:\n{args.get('text') or ''}",
        )

    def _text_generation(self, args):
        return self._llm_text(
            "You are a text generation tool. Fulfil the prompt directly and "
            "concisely; return only the generated text.",
            str(args.get("text") or args.get("prompt") or ""),
        )

    def _question_answering(self, args):
        passage = str(args.get("text") or "")
        question = str(args.get("question") or "")
        return self._llm_text(
            "You are a question-answering tool. Answer briefly; if a passage "
            "is given, ground the answer in it.",
            f"Passage:\n{passage}\n\nQuestion: {question}",
        )

    def _wikipedia_search(self, args):
        topic = str(args.get("text") or args.get("query") or "")
        url = (
            "https://simple.wikipedia.org/w/api.php?action=query&prop=extracts"
            "&exintro=1&explaintext=1&format=json&redirects=1&titles="
            + urllib.parse.quote(topic)
        )
        data = json.loads(self._http_get(url))
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            extract = (page.get("extract") or "").strip()
            if extract:
                return extract[:1200]
        return f"(no Simple Wikipedia page found for '{topic}')"

    def _numbers_fact(self, kind):
        # numbersapi.com no longer serves the plain-text API (404 as of
        # 2026-07); back these tools with the serving LLM instead — a
        # documented substitution consistent with the other LLM-backed tools.
        def handler(self, args):
            subject = (args.get("number") or args.get("year")
                       or args.get("date") or "")
            return self._llm_text(
                f"You are a '{kind} fact' tool. Return exactly one short, "
                "true, interesting fact. No preamble.",
                f"Give a {kind} fact about: {subject}",
            )

        return handler

    _HANDLERS = {
        # GTA
        "OCR": _ocr_tool,
        "ImageDescription": _image_description,
        "CountGivenObject": _count_given_object,
        "Calculator": _calculator,
        "Solver": _solver,
        "GoogleSearch": _google_search,
        # m&m
        "text classification": _text_classification,
        "text summarization": _text_summarization,
        "text generation": _text_generation,
        "question answering": _question_answering,
        "wikipedia simple search": _wikipedia_search,
    }


# numbersapi handlers need the factory; attach after class body
for _kind, _name in [("math", "get math fact"), ("trivia", "get trivia fact"),
                     ("year", "get year fact"), ("date", "get date fact")]:
    RealToolExecutor._HANDLERS[_name] = RealToolExecutor._numbers_fact(None, _kind)
