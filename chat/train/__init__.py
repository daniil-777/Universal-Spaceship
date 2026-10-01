"""chat/train — CAPCOM's training stages (spec §6): S1 teacher SFT, S2 student SFT (sft_stage), S3 on-policy distillation
(distill_stage), S4 multi-turn-aware DPO (dpo_stage), S5 optional GRPO (grpo_stage), S6 pick best + merge (merge_stage), S7 web
export (web_stage). Shared plumbing (GPU tier, models, hyper-parameters, data, DONE markers, Drive mirroring) lives in common."""
