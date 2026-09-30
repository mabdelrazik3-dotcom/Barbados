"""External line readers (the diagram's "PP-OCR" reader slot). Each maps a line image to one text.

    paddleocr  PaddleOCR text recognition (3.x `TextRecognition`, or the 2.x `PaddleOCR(...).ocr`)
    trocr      a transformers VisionEncoderDecoder checkpoint (e.g. microsoft/trocr-large-handwritten)
    csv        predictions produced elsewhere: a CSV with columns ID,text

Their outputs join the candidate pool and give agreement features (loss to the reader's text).
"""
from __future__ import annotations

import pandas as pd

from ..config import Cfg
from ..text.clean import clean_line
from ..utils import log


class ExternalReader:
    def __init__(self, spec: dict, cfg: Cfg):
        self.spec, self.cfg, self.name = spec, cfg, spec["name"]
        kind = spec["kind"]
        if kind == "paddleocr":
            try:
                from paddleocr import TextRecognition

                self._rec = TextRecognition(model_name=spec.get("model_name", "en_PP-OCRv5_mobile_rec"))
                self._predict = self._paddle3
            except ImportError:
                from paddleocr import PaddleOCR

                self._rec = PaddleOCR(use_angle_cls=False, lang=spec.get("lang", "en"))
                self._predict = self._paddle2
        elif kind == "trocr":
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            self._torch = torch
            self._proc = TrOCRProcessor.from_pretrained(spec["hf_id"])
            self._model = VisionEncoderDecoderModel.from_pretrained(spec["hf_id"]).eval()
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            self._predict = self._trocr
        elif kind == "csv":
            df = pd.read_csv(spec["path"], dtype=str, keep_default_na=False)
            self._table = dict(zip(df.ID, df.text))
            self._predict = lambda ids, paths: [self._table.get(i, "") for i in ids]
        else:
            raise ValueError(f"unknown external reader kind '{kind}'")

    def _paddle3(self, ids, paths):
        out = []
        for p in paths:
            res = list(self._rec.predict(input=p))
            out.append(str(res[0]["rec_text"]) if res else "")
        return out

    def _paddle2(self, ids, paths):
        out = []
        for p in paths:
            res = self._rec.ocr(p, det=False, cls=False)
            out.append(str(res[0][0][0]) if res and res[0] else "")
        return out

    def _trocr(self, ids, paths):
        from PIL import Image

        out = []
        for p in paths:
            with Image.open(p) as im:
                pixel_values = self._proc(images=im.convert("RGB"), return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self._model.device)
            with self._torch.no_grad():
                gen = self._model.generate(pixel_values, max_new_tokens=128)
            out.append(self._proc.batch_decode(gen, skip_special_tokens=True)[0])
        return out

    def predict(self, rows: pd.DataFrame) -> dict[str, str]:
        texts = self._predict(list(rows.ID), list(rows.image_path))
        log.info("external reader %s: %d lines", self.name, len(texts))
        return {i: clean_line(t) for i, t in zip(rows.ID, texts)}
