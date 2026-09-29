"""LoRA supervised fine-tuning of Qwen-VL models on line crops.

Backends
    hf       transformers Trainer + peft LoRA (optionally QLoRA with 4-bit weights)
    unsloth  the 1st place's recipe: FastVisionModel + UnslothVisionDataCollator + TRL SFTTrainer
    mock     writes the adapter metadata only (dry runs)

Every adapter directory gets `brainiac_meta.json` (backbone, prompt spec, view spec, answer
format). Inference reads it back, so a model is always prompted exactly as it was trained.
"""
from __future__ import annotations

from pathlib import Path

from ..config import Cfg, preset
from ..data.views import ViewSpec, get_views, view_instruction
from ..text.prompts import build_messages, build_prompt
from ..utils import free_gpu, load_json, log, save_json, set_seed
from .registry import get_spec, load_model, load_processor, model_source

LANGUAGE_TARGETS = r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
META = "brainiac_meta.json"


def adapter_ready(path: str | Path) -> bool:
    path = Path(path)
    if not (path / META).exists():
        return False
    return load_json(path / META).get("backend") == "mock" or (path / "adapter_config.json").exists()


def adapter_meta(path: str | Path) -> dict:
    return load_json(Path(path) / META)


class LineSFTDataset:
    """Lazy dataset: views are loaded per item, so training never holds every image in memory."""

    def __init__(self, cfg: Cfg, examples: list[dict], view: ViewSpec, prompt: str, instruction: str,
                 as_messages: bool = False):
        self.cfg, self.examples, self.view = cfg, examples, view
        self.prompt, self.instruction, self.as_messages = prompt, instruction, as_messages

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        images = get_views(self.cfg, self.view, ex["ID"], ex["image_path"], ex["family"])
        if not self.as_messages:
            return {"images": images, "answer": ex["answer"]}
        content = [{"type": "text", "text": self.prompt}]
        content += [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": self.instruction})
        return {"messages": [{"role": "user", "content": content},
                             {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]}]}


class SFTCollator:
    """Chat-formats a batch and masks everything but the answer (+ end marker) in the labels."""

    def __init__(self, processor, prompt: str, instruction: str, end_marker: str):
        self.processor, self.tok = processor, processor.tokenizer
        self.prompt, self.instruction, self.end_marker = prompt, instruction, end_marker

    def __call__(self, batch: list[dict]) -> dict:
        import torch

        texts, images, answer_ids = [], [], []
        for item in batch:
            msgs = build_messages(self.prompt, len(item["images"]), self.instruction)
            prefix = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            texts.append(prefix + item["answer"] + self.end_marker)
            images.extend(item["images"])
            answer_ids.append(self.tok(item["answer"] + self.end_marker, add_special_tokens=False)["input_ids"])
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = torch.full_like(enc["input_ids"], -100)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        for b, ids in enumerate(answer_ids):
            end, n = int(lengths[b]), len(ids)
            if enc["input_ids"][b, end - n : end].tolist() != ids:
                n += 1  # the answer's first token merged with the prompt's last one
            labels[b, end - n : end] = enc["input_ids"][b, end - n : end]
        enc["labels"] = labels
        return enc


def lora_config(lp: Cfg):
    from peft import LoraConfig

    targets = lp.targets
    if targets == "language":
        targets = LANGUAGE_TARGETS
    elif targets == "all_linear":
        targets = "all-linear"
    return LoraConfig(r=int(lp.r), lora_alpha=int(lp.alpha), lora_dropout=float(lp.dropout), target_modules=targets,
                      bias="none", task_type="CAUSAL_LM", use_rslora=bool(lp.get("use_rslora", False)))


def _training_args(tp: Cfg, out_dir: Path, seed: int, cuda: bool) -> dict:
    return dict(
        output_dir=str(out_dir / "trainer_state"),
        per_device_train_batch_size=int(tp.batch_size),
        gradient_accumulation_steps=int(tp.grad_accum),
        learning_rate=float(tp.lr),
        num_train_epochs=float(tp.epochs),
        max_steps=int(tp.max_steps),
        warmup_ratio=float(tp.warmup_ratio),
        lr_scheduler_type=str(tp.scheduler),
        weight_decay=float(tp.weight_decay),
        max_grad_norm=float(tp.max_grad_norm),
        bf16=bool(tp.bf16) and cuda,
        logging_steps=int(tp.logging_steps),
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        optim=str(tp.optim),
        neftune_noise_alpha=float(tp.neftune_noise_alpha) if tp.get("neftune_noise_alpha") else None,
        remove_unused_columns=False,
        dataloader_num_workers=int(tp.num_workers),
    )


def _train_hf(cfg, spec, lp, tp, dataset, prompt, instruction, out_dir: Path, seed: int) -> None:
    import torch
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    processor = load_processor(spec, cfg, padding_side="right")
    model = load_model(spec, cfg, training=True)
    model.config.use_cache = False
    if spec.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(tp.gradient_checkpointing))
    elif tp.gradient_checkpointing:
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(lp))
    model.print_trainable_parameters()
    args = _training_args(tp, out_dir, seed, torch.cuda.is_available())
    args.update(gradient_checkpointing=bool(tp.gradient_checkpointing),
                gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=TrainingArguments(**args), train_dataset=dataset,
                      data_collator=SFTCollator(processor, prompt, instruction, spec.end_marker))
    trainer.train()
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    del trainer, model
    free_gpu()


def _train_unsloth(cfg, spec, lp, tp, dataset, out_dir: Path, seed: int) -> None:
    """The 1st place's training call (unsloth 2025.10 / trl 0.22), with our data and settings."""
    import os

    os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")
    from unsloth import FastVisionModel  # noqa: I001 - unsloth must be imported before trl/transformers
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    source = model_source(spec, unsloth=True)
    model, tokenizer = FastVisionModel.from_pretrained(
        source, load_in_4bit=bool(spec.load_in_4bit or "bnb-4bit" in source), use_gradient_checkpointing="unsloth"
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lp.targets == "all_linear",
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=int(lp.r),
        lora_alpha=int(lp.alpha),
        lora_dropout=float(lp.dropout),
        bias="none",
        random_state=seed,
        use_rslora=bool(lp.get("use_rslora", False)),
        loftq_config=None,
    )
    FastVisionModel.for_training(model)
    args = _training_args(tp, out_dir, seed, True)
    args.update(gradient_checkpointing=bool(tp.gradient_checkpointing), tf32=True, dataset_text_field=None,
                dataset_kwargs={"skip_prepare_dataset": True}, packing=False)
    trainer = SFTTrainer(model=model, data_collator=UnslothVisionDataCollator(model, tokenizer),
                         train_dataset=dataset, args=SFTConfig(**args))
    trainer.train()
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    del trainer, model
    free_gpu()


def train_lora(cfg: Cfg, *, model_name: str, lora: str | dict, train: str | dict, examples: list[dict],
               out_dir: str | Path, seed: int, prompt_spec: dict, view_name: str, dual: bool = False,
               dual_order: str = "ink_first", force: bool = False) -> Path:
    """Train one LoRA adapter. `examples`: dicts with ID, image_path, family, answer (JSON text)."""
    out_dir = Path(out_dir)
    if adapter_ready(out_dir) and not force:
        log.info("adapter exists: %s", out_dir)
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = get_spec(cfg, model_name)
    lp, tp = preset(cfg, "lora_presets", lora), preset(cfg, "train_presets", train)
    view = ViewSpec.from_cfg(cfg, view_name)
    prompt = build_prompt(cfg, prompt_spec, dual=dual, dual_order=dual_order)
    instruction = view_instruction(view)
    set_seed(seed, bool(cfg.deterministic))
    log.info("train %s [%s, %s] on %d examples -> %s", model_name, tp.backend, view_name, len(examples), out_dir)
    if tp.backend == "hf":
        _train_hf(cfg, spec, lp, tp, LineSFTDataset(cfg, examples, view, prompt, instruction), prompt, instruction,
                  out_dir, seed)
    elif tp.backend == "unsloth":
        _train_unsloth(cfg, spec, lp, tp, LineSFTDataset(cfg, examples, view, prompt, instruction, as_messages=True),
                       out_dir, seed)
    elif tp.backend != "mock":
        raise ValueError(f"unknown training backend '{tp.backend}'")
    save_json({"model": model_name, "hf_id": spec.hf_id, "backend": tp.backend, "prompt": dict(prompt_spec),
               "view": view_name, "dual": dual, "dual_order": dual_order, "lora": dict(lp), "train": dict(tp),
               "seed": seed, "n_examples": len(examples)}, out_dir / META)
    return out_dir
