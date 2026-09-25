// Full-screen colour pass applied after tone mapping: mode 0 passthrough, 1 mono (greyscale with a gentle S-curve),
// 2 ink (true two-tone: luminance vs an 8×8 Bayer ordered-dither threshold, white on black or inverted).
export const MonoShader = {
  name: 'MonoShader',
  uniforms: { tDiffuse: { value: null }, mode: { value: 0 }, invert: { value: 0 }, cell: { value: 2 }, contrast: { value: 0 } },   // contrast 0..1: extra tonal separation for bright daylight scenes
  vertexShader: /* glsl */`
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */`
    uniform sampler2D tDiffuse; uniform int mode; uniform float invert; uniform float cell, contrast;
    varying vec2 vUv;
    float b2(vec2 p) { p = mod(floor(p), 2.0); return mod(2.0 * p.x + 3.0 * p.y, 4.0); }           // 2×2 Bayer: [[0,2],[3,1]]
    float bayer8(vec2 p) { return (16.0 * b2(p) + 4.0 * b2(floor(p) / 2.0) + b2(floor(p) / 4.0) + 0.5) / 64.0; }
    void main() {
      vec4 c = texture2D(tDiffuse, vUv);
      if (mode == 0) { gl_FragColor = c; return; }
      float l = dot(c.rgb, vec3(0.2126, 0.7152, 0.0722));
      if (mode == 1) { float g = l * l * (3.0 - 2.0 * l); g = mix(l, g, 0.55); gl_FragColor = vec4(vec3(mix(g, 1.0 - g, invert)), 1.0); return; }
      float t = bayer8(floor(gl_FragCoord.xy / cell));
      float g = pow(max(0.0, (l - 0.045) / 0.955), 0.8);      // black point: the faint sky stays solid black, only lit surfaces dither
      g = mix(g, smoothstep(0.16, 0.84, g), contrast);       // daylight: push the mid-greys apart so sky and snow read white, sea and shadow black
      float o = g > t ? 1.0 : 0.0;
      gl_FragColor = vec4(vec3(mix(o, 1.0 - o, invert)), 1.0);
    }`,
};
