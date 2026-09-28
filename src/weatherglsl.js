// GLSL mirror of src/weather.js's cellShape(), rainShape(), lobeG() and cellLobed() for the cloud renderer and the cloud-shadow map. Cells travel
// to the GPU as a float texture, two texels per cell: (x, z, R, type) and (base, top, sig, seed), x in render space.
// Keep in step with weather.js — the harness's weather_match.mjs compares the two at 1000 random points.
export const CELL_GLSL = /* glsl */`
  float wSmooth(float a, float b, float x) { float t = clamp((x - a) / (b - a), 0.0, 1.0); return t * t * (3.0 - 2.0 * t); }
  float cellShape(float type, float R, float base, float top, float dx, float y, float dz) {
    float h = (y - base) / (top - base); if (h < 0.0 || h > 1.0) return 0.0;
    float ez = type > 1.5 ? dz * 2.0 : dz, r = length(vec2(dx, ez)) / R, shrink = type > 1.5 ? 1.0 : 1.0 - 0.55 * h * h, rr = r / max(shrink, 0.05);
    return (1.0 - wSmooth(0.6, 1.0, rr)) * wSmooth(0.0, 0.06, h) * (1.0 - wSmooth(0.85, 1.0, h));
  }
  float rainShape(float type, float R, float base, float groundY, float dx, float y, float dz) {
    if (type < 0.5 || type > 1.5 || y > base || y < groundY) return 0.0;
    float r = length(vec2(dx, dz)) / (0.5 * R); return (1.0 - wSmooth(0.5, 1.0, r)) * wSmooth(groundY, groundY + 4.0, y);
  }
  float lobeG(float r, float amount, float median) { return (clamp((r - 0.3) / 0.62, 0.0, 1.0) - median) * amount; }   // weather.js LOBE / lobeG
  float cellLobed(float type, float R, float base, float top, float dx, float y, float dz, float g) {
    bool st = type > 1.5; float gi = st ? g * 0.7 : g; return cellShape(type, R * (1.0 + gi), base + (st ? 0.0 : gi * 0.8), top + gi * (st ? 0.15 : 0.3) * (top - base), dx, y, dz);
  }`;
