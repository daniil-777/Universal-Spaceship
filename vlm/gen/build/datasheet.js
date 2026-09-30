// vlm/gen/build/datasheet.js — Gebru et al.'s 7 sections, auto-filled from stats.json and the build metadata (spec §9), with
// the records amendment B asks for: the Pilot Eye scope removals (EYE.hidden, closingRhoM), the D contact-range exclusion
// (T10-g), the Z capture protocol (T9-a), the EOX budget per run (T9-b), the visibility-gate limitations (T10v / T10-j) and
// the verifier's free-text limitations with their measured rates (T7-d). The gate constants are quoted from
// vlm/gen/labels/landing.js (T_VIS) and corridor_facts.js (IN_CLOUD_A, IN_CLOUD_VIS_U).
const J = (x) => JSON.stringify(x), SPEC32 = Object.freeze({ Z: 800, S: 1200, A: 1600, L: 800, D: 600 });
const table = (o) => (Object.keys(o || {}).length ? '| key | value |\n|---|---|\n' + Object.entries(o).map(([k, v]) => `| ${k} | ${typeof v === 'object' ? J(v) : v} |`).join('\n') : '(none)');
const pct = (x) => (x === null || x === undefined ? 'n/a' : `${(100 * x).toFixed(2)} %`);
export function datasheet(s, m) {
  const t = s.text || {}, fam = t.byFamily || {}, eye = m.eye, z = s.z || { range_bins: {}, discarded_views: [] }, a = s.a || {};
  const famRates = Object.entries(fam).map(([f, x]) => `${f} ${pct(x.rejectRate)} (${x.rejected}/${x.total})`).join(', ') || 'n/a';
  return [
    `# Datasheet: ${m.name}`,
    `Built ${m.date} by vlm/build.mjs at ${m.build_sha || 'unknown'} from runs ${m.runs.join(', ')} (captured at ${m.git_sha}); licence profile ${m.licence}; capture_mode ${m.capture_mode}.`,
    '## 1 Motivation',
    'Training and evaluating two in-browser models for Astro Pilot: Pilot Eye (a per-frame safety monitor on 3 frames at 160x96) and Narrator (a SmolVLM-256M captioner and VQA model with a Context line). Created by the Astro Pilot project.',
    '## 2 Composition',
    'Records per family | split | safety_eye verdict (Z: zoom):', table(s.counts),
    `Sizes against spec §3.2 (R17; v0 reports them and does not apply the §11.3 size rule, ruling T3H-1): records built per family ${J(m.captured || {})} of §3.2 ${J(SPEC32)}; planned sizes from the G4 projection ${J(m.sizes)}.`,
    `Natural verdict mix per family x world (n, and summed sampler_weight): ${J(s.naturalMix)}.`,
    `A keeps its natural, UNSAFE-heavy mix (ruling T10-h; the class-balanced weights balance it at sampling time): raw ${J(a.raw_mix)}, natural sampler-weighted ${J(a.natural_mix_sampler_weighted)}; per route x behaviour ${J(a.by_route_policy)}.`,
    `Minimal pairs: ${J(s.pairs)}. Pixel-identical pairs are kept for training and Narrator but excluded from the paired-verdict metric.`,
    `Z views per split x range bin (edges 20/100/400/1500 km): ${J(z.range_bins)}. Val and test are expected to be thin in the > 1500 km bin (a wide footprint rarely stays in one split). Discarded Z views (split re-check on the 32x18 grid widened by one cell): ${z.discarded_views.length}.`,
    `Pilot Eye export: ${J(s.eye)}. D records with axial < 0.5 m are excluded from the Pilot Eye export and cache (the untextured port face fills the eye view; ruling T10-g) and kept for Narrator rows, which use the chase frame.`,
    `Dedupe drops (64-bit dHash of f2, cross-split Hamming <= 4, dropped from val/test/ood): ${J(s.dedupeDrops)}. Invalid or text-fact-failing records left out: ${s.rejectedRecords ?? 0}.`,
    '## 3 Collection process',
    `Frames rendered by the site's own code in headless Chromium (Playwright ${m.playwright}), one fresh page per episode, a paused fake clock and a seeded Math.random, at 896x504, renderScale 1, dpr 1. Labels computed in the page from the rendered objects. Policies: S belt_ppo, A atmo_ppo or the search pilot (search_v1), L autoland, D GNC.`,
    `A samples continue across the page's own crash resets (each reset is a new episode index); a post-reset sample within 4.6 s is accepted with p = 1/3 and weight 3 (max 1 per segment), and twins carry their original's sampler_weight. A pages ${a.pages ?? 0}, resets seen ${a.resets_seen ?? 0}, records by segment ${J(a.records_by_segment)}, t_since_reset_s histogram ${J(a.t_since_reset_hist)}.`,
    'Z capture protocol (ruling T9-a): the page clock is frozen while any tile request or zoom-loader job is pending, each view settles until the rendered state is stable, and the capture is taken at the end of a fixed 1600-frame slot of page time, so a cold and a warm run capture the same state.',
    `Imagery budget (ruling T9-b): at most 80,000 EOX requests per run (the used count persists in raw/<run>/eox_budget.json). Upstream requests and cache hits per host: ${J(s.upstream)}.`,
    '## 4 Preprocessing, cleaning, labelling',
    `Safety labels by simulated branches (spec §4.3): corridor HANDS_OFF reference, H = 3 s, K = 4 draws, pulses 8/15 steps then hands off; landing autoland and docking GNC to the end of the run. c_near = ${m.c_near} u. D records whose continued run ends in fail:timeout or fail:keep_in are dropped at capture (ruling T4-d).`,
    `safety_eye (Pilot Eye's scope) = the same rules without what Pilot Eye cannot see at 160x96: EYE.hidden ${eye.hidden.join(', ')}; closingRhoM ${eye.closingRhoM} (m); locNm ${eye.locNm} (NM); removed criteria ${J(m.eye_removed || {})}. labels.json lists the same.`,
    'Visibility gate (T10v, ruling T10-j): visual facts of an object are null when fog or cloud hides it. L uses exp² scene fog with the transmittance cutoff T_VIS 0.1; A uses the linear scene fog with the same cutoff plus the in-cloud rule (air.in_cloud >= 0.5 and distance > IN_CLOUD_VIS_U = 12 u). L captions and VQA use scene.in_cloud for frames inside the cloud deck. Limitations: the 12 u cutoff is measured on 5 frames; cloud density is sampled at the ship\'s position, not along each object\'s ray; the linear A fog is inert on the current routes.',
    `Text: template output from the paraphrase bank, each item checked by the §5.4 verifier (failures rejected, never repaired); the teacher was off. Verifier rejection rate ${pct(t.rejectRate)} (${t.rejected ?? 0} of ${(t.items ?? 0) + (t.rejected ?? 0)}), per family ${famRates}; parser rejections ${pct(t.parserFalseReject)}; hand-checked false rejects: see the build report. The verifier is authoritative for template text and a conservative screen for free text (ruling T7-d); its measured free-text limitations: verdict synonyms without a verdict word are not claims (catch 0 %); comparatives with an earlier moment or another object (0 %); superlatives, "only" and "both" (20 %); binding of size, body-closing and place-distance claims to their subject (67 / 47 / 73 % template / free / Narrator); the teacher path lacks vert_mode and wow; double negation is resolved by parity within a clause only (128/128 in-clause). The §11.5 hallucination metric is therefore a lower bound.`,
    'Splits: S/A by episode seed, L/D by run seed (mod 10: 0-7 train, 8 val, 9 test), Z by level-3 block from the plan; OOD: A moscow and A cloudy, D far, the Z Oceania/New Guinea region; twins stay in their original\'s split. 20 % of paraphrase ids are held out for test and OOD. VQA answers are balanced by rejection sampling per question family (<= 50 % per answer). Pilot Eye weights are class-balanced per family x verdict; sampler_weight is stored so natural rates can be recovered.',
    '## 5 Uses',
    'Intended: research and demos of in-browser vision models on simulator frames. Not for real flight, landing or docking decisions.',
    '## 6 Distribution',
    `Attribution: see ATTRIBUTION.txt. Imagery licence profile ${m.licence}. No teacher (paid API) outputs are included.`,
    '## 7 Maintenance',
    'Maintained in the Astro Pilot repository (vlm/). Known gaps: procedural flight scenery (air captions name regions only), the sim-to-real gap, pilot bias in the behaviour policies (A samples cluster after crash resets), EOX 2016 cloud and age, Earth visibility in S is not labelled (no named Earth mesh), and the visibility-gate limitations above.',
  ].join('\n\n') + '\n';
}
