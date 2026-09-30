// vlm/web/narrator.js — the on-demand Narrator (spec §10.3, controller amendment A): transformers.js 4.3 (injected), WebGPU with
// a WASM fallback, the deploy dtypes {embed_tokens fp16, vision_encoder q8 (our weight-only 8-bit re-quantization),
// decoder_model_merged q4f16 (the fused GenAI-builder decoder)}; G6 may pick decoder 'q4'. The exported preprocessor has
// do_resize off, so every frame is first squared to 512x512 (Task 8's square(): LANCZOS) and handed to
// processor(text, [image], {do_image_splitting: false}); the §5.7 Context line (Pilot Eye's heads + the game's telemetry)
// goes before the task on its own line, as in the training rows; greedy, streaming until eos 49279.
// model 'lfm' = Plan B (§10.4): its own processor and a single q4f16 dtype.
import { renderContext, telemetryOf, ttcBin, clrBin } from '../gen/text/context.js';
export const NARRATOR_EOS = 49279;
export const NARRATOR_EDGE = 512;
// the demo's buttons: the trained caption_detail and safety prompts (apv-pilot narrator train: 224 and 180 rows)
export const NARRATOR_TASKS = Object.freeze({ describe: 'Describe the image in detail.', safety: 'Is the situation safe? Explain.' });
// the frame the Narrator was trained on for a Pilot Eye row: the Narrator row's images[0] joined on key (images: Map key -> path)
// when known; else D's chase view (<key>.chase.png beside the port frames); else the last Pilot Eye frame (S/A/L/Z records have
// narrator_frame == frames[-1])
export function narratorFrame(row, images = null) {
  const hit = images && images.get(row.key), last = row.frames[row.frames.length - 1];
  return hit || (row.family === 'D' ? last.replace(/[^/]*$/, `${row.key}.chase.png`) : last);
}
const p2 = (x) => +(+x).toFixed(2);
// the Monitor tuple of the Context line (the Task 7 shape; evaluate.py monitor(..., allowed=reason_masks[family]) writes the same
// from the PyTorch heads: createPilotEye filters h.reasons by the same labels.json reason_masks)
export function monitorFromHeads(h, family) {
  const sa = family === 'S' || family === 'A';
  return { verdict: h.verdict, severity: h.severity, reasons: [...h.reasons], action: h.action, p_ref: sa ? p2(h.p_ref) : null, ttc_bin: sa ? ttcBin(h.ttc_s) : 'none', clr_bin: sa ? clrBin(h.clearance_u) : 'none' };
}
export const contextLine = ({ heads = null, facts = {}, family }) => renderContext({ telemetry: telemetryOf(facts, family), monitor: heads && family !== 'Z' ? monitorFromHeads(heads, family) : null });
// Task 8's square(): the one resize before the processor. Node (sharp) resizes with LANCZOS (resample 1); in a page,
// transformers.js resizes through a canvas without a filter choice, so a page draws with imageSmoothingQuality 'high'.
export async function squareImage(tf, image, edge = NARRATOR_EDGE) {
  if (image.width === edge && image.height === edge) return image;
  if (typeof OffscreenCanvas !== 'undefined' && typeof image.toCanvas === 'function' && tf.RawImage) {
    const c = new OffscreenCanvas(edge, edge), g = c.getContext('2d'); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high';
    g.drawImage(image.toCanvas(), 0, 0, edge, edge);
    return new tf.RawImage(g.getImageData(0, 0, edge, edge).data, edge, edge, 4).rgb();
  }
  return image.resize(edge, edge, { resample: 1 });
}
export async function createNarrator({ tf, modelId, localModelPath = null, device = 'webgpu', fallback = 'wasm', model = 'smolvlm', decoder = 'q4f16', vision = 'q8' }) {
  if (localModelPath) { tf.env.localModelPath = localModelPath; tf.env.allowRemoteModels = false; tf.env.allowLocalModels = true; }
  const dtype = model === 'lfm' ? 'q4f16' : { embed_tokens: 'fp16', vision_encoder: vision, decoder_model_merged: decoder }, t0 = performance.now();
  const processor = await tf.AutoProcessor.from_pretrained(modelId); let mdl, usedDevice = device;
  try { mdl = await tf.AutoModelForVision2Seq.from_pretrained(modelId, { device, dtype }); } catch (e) {
    if (!fallback || fallback === device) throw e;
    usedDevice = fallback; mdl = await tf.AutoModelForVision2Seq.from_pretrained(modelId, { device: fallback, dtype });
  }
  const loadMs = performance.now() - t0;
  return {
    loadMs, device: usedDevice, dtype, model: mdl, processor,
    async describe(image, { context = null, task = NARRATOR_TASKS.describe, maxNewTokens = 90, minNewTokens = 0, onToken = null } = {}) {
      const t1 = performance.now(), img = model === 'lfm' ? image : await squareImage(tf, image);
      const text = processor.apply_chat_template([{ role: 'user', content: [{ type: 'image' }, { type: 'text', text: context ? `${context}\n${task}` : task }] }], { add_generation_prompt: true });
      const inputs = model === 'lfm' ? await processor(text, [img]) : await processor(text, [img], { do_image_splitting: false });
      const streamer = onToken ? new tf.TextStreamer(processor.tokenizer, { skip_prompt: true, skip_special_tokens: true, callback_function: onToken }) : undefined;
      const ids = await mdl.generate({ ...inputs, max_new_tokens: maxNewTokens, min_new_tokens: minNewTokens, do_sample: false, eos_token_id: NARRATOR_EOS, streamer });
      return { text: processor.batch_decode(ids.slice(null, [inputs.input_ids.dims.at(-1), null]), { skip_special_tokens: true })[0].trim(), ms: performance.now() - t1 };
    },
    dispose: () => (mdl.dispose ? mdl.dispose() : undefined),
  };
}
