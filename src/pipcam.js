// Onboard camera: a second, small view of the scene from the ship — NOSE (forward, banking with the ship), BELLY (down
// and a little ahead, horizon-stable) or TOP (a moving map from above, heading up) — rendered at 30 Hz into its own
// half-float target (supersampled 1.5×, no MSAA: multisampled targets render black bands on ANGLE/Metal) and composited
// every frame into the main canvas under the flight board's frame. The composite applies the page's tone mapping and
// colour mode (mono S-curve / ink Bayer dither, as in post.js), rounded corners, a soft vignette and faint scan lines.
import * as THREE from 'three';

export const PIP_MODES = [
  { key: 'nose', name: 'NOSE', fov: 66 },
  { key: 'belly', name: 'BELLY', fov: 60 },
  { key: 'top', name: 'TOP', fov: 42 },
];

const FRAG = /* glsl */`
  uniform sampler2D tPip; uniform vec2 uSize; uniform float uRadius, uCell, uContrast, uInvert; uniform int uMode;
  varying vec2 vUv;
  float b2(vec2 p) { p = mod(floor(p), 2.0); return mod(2.0 * p.x + 3.0 * p.y, 4.0); }
  float bayer8(vec2 p) { return (16.0 * b2(p) + 4.0 * b2(floor(p) / 2.0) + b2(floor(p) / 4.0) + 0.5) / 64.0; }
  void main() {
    vec2 p = (vUv - 0.5) * uSize, q = abs(p) - (0.5 * uSize - uRadius);
    float d = length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - uRadius, a = 1.0 - smoothstep(-0.75, 0.75, d);   // rounded rectangle, ~1.5 px soft edge
    vec3 c = texture2D(tPip, vUv).rgb;
    #ifdef TONE_MAPPING
      c = toneMapping(c);
    #endif
    c = linearToOutputTexel(vec4(c, 1.0)).rgb;                                   // display-referred, like the main image after OutputPass
    if (uMode == 0) {
      vec2 e = vUv - 0.5; c *= 1.0 - 0.55 * dot(e, e);                           // soft vignette
      c *= 0.965 + 0.035 * step(0.5, fract(gl_FragCoord.y / 3.0));               // faint scan lines
    } else {
      float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
      if (uMode == 1) { float g = l * l * (3.0 - 2.0 * l); g = mix(l, g, 0.55); vec2 e = vUv - 0.5; g *= 1.0 - 0.45 * dot(e, e); c = vec3(mix(g, 1.0 - g, uInvert)); }
      else {
        float t = bayer8(floor(gl_FragCoord.xy / uCell)), g = pow(max(0.0, (l - 0.045) / 0.955), 0.8);
        g = mix(g, smoothstep(0.16, 0.84, g), uContrast); float o = g > t ? 1.0 : 0.0; c = vec3(mix(o, 1.0 - o, uInvert));
      }
    }
    gl_FragColor = vec4(c, a);
  }`;
const VERT = /* glsl */`varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

export function createPipCam(renderer) {
  const cam = new THREE.PerspectiveCamera(66, 16 / 10, 0.25, 9000);
  const rt = new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, depthBuffer: true });
  rt.texture.minFilter = THREE.LinearFilter; rt.texture.magFilter = THREE.LinearFilter; rt.texture.generateMipmaps = false;
  const mat = new THREE.ShaderMaterial({
    vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthTest: false, depthWrite: false,
    uniforms: { tPip: { value: rt.texture }, uSize: { value: new THREE.Vector2(1, 1) }, uRadius: { value: 13 }, uMode: { value: 0 }, uInvert: { value: 0 }, uCell: { value: 2 }, uContrast: { value: 0 } },
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat); quad.frustumCulled = false;
  const qScene = new THREE.Scene(); qScene.add(quad); const qCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const size = new THREE.Vector2(), look = new THREE.Vector3(), fh = new THREE.Vector3(), tmp = new THREE.Vector3(), WORLD_UP = new THREE.Vector3(0, 1, 0);
  let mode = 0, rect = null, acc = 1, fresh = false;

  function pose(ship) {                                   // ship: { p, q, f, u } (render space)
    const m = PIP_MODES[mode];
    if (cam.fov !== m.fov) { cam.fov = m.fov; cam.updateProjectionMatrix(); }
    fh.set(ship.f.x, 0, ship.f.z); if (fh.lengthSq() < 1e-6) fh.set(1, 0, 0); fh.normalize();   // heading, level
    if (m.key === 'nose') {                               // just ahead of the nose, looking where the ship points, banking with it
      cam.position.copy(ship.p).addScaledVector(ship.f, 2.1).addScaledVector(ship.u, 0.3); cam.up.copy(ship.u); cam.lookAt(look.copy(cam.position).add(ship.f));
    } else if (m.key === 'belly') {                       // under the hull, looking down and 30° ahead, horizon-stable
      cam.position.copy(ship.p).addScaledVector(WORLD_UP, -1.3); cam.up.copy(fh); cam.lookAt(look.copy(cam.position).addScaledVector(WORLD_UP, -1).addScaledVector(fh, 0.58));
    } else {                                              // a moving map: straight down from 58 units, heading up
      cam.position.copy(ship.p).addScaledVector(WORLD_UP, 58); cam.up.copy(fh); cam.lookAt(ship.p);
    }
    cam.updateMatrixWorld();
  }

  return {
    get mode() { return PIP_MODES[mode]; },
    cycle() { mode = (mode + 1) % PIP_MODES.length; acc = 1; return PIP_MODES[mode]; },
    setMode(key) { const i = PIP_MODES.findIndex((m) => m.key === key); if (i >= 0) { mode = i; acc = 1; } },
    setRect(r) { rect = r && r.w > 8 && r.h > 8 ? r : null; acc = 1; fresh = false; },   // CSS px, top-left origin; null = hidden
    get active() { return !!rect; },
    // hide: objects to hide in the nose and belly views (the ship itself, sensor beams…); style: the main image's colour mode
    render(scene, ship, dt, { hide = [], hideAlways = [], style = { mode: 0, invert: 0, cell: 2, contrast: 0 }, fps = 60 } = {}) {
      if (!rect) return;
      renderer.getSize(size);
      acc += dt;
      if (acc >= (fps < 52 ? 1 / 15 : 1 / 30) || !fresh) {   // 30 Hz is plenty for a picture this small (15 Hz when the frame rate sags); the composite below runs every frame
        acc = 0; fresh = true;
        const pr = renderer.getPixelRatio(), w = Math.min(560, Math.round(rect.w * pr * 1.5)), h = Math.min(400, Math.round(rect.h * pr * 1.5));
        if (rt.width !== w || rt.height !== h) rt.setSize(w, h);
        if (Math.abs(cam.aspect - rect.w / rect.h) > 1e-3) { cam.aspect = rect.w / rect.h; cam.updateProjectionMatrix(); }
        pose(ship);
        const inside = PIP_MODES[mode].key !== 'top', hidden = [];
        for (const o of inside ? hide.concat(hideAlways) : hideAlways) if (o && o.visible) { o.visible = false; hidden.push(o); }
        const prev = renderer.getRenderTarget(); renderer.setRenderTarget(rt); renderer.clear(); renderer.render(scene, cam); renderer.setRenderTarget(prev);
        for (const o of hidden) o.visible = true;
      }
      const U = mat.uniforms; U.uSize.value.set(rect.w, rect.h); U.uMode.value = style.mode; U.uInvert.value = style.invert; U.uCell.value = style.cell; U.uContrast.value = style.contrast;
      const y = size.y - rect.y - rect.h, auto = renderer.autoClear;
      renderer.setViewport(rect.x, y, rect.w, rect.h); renderer.setScissor(rect.x, y, rect.w, rect.h); renderer.setScissorTest(true);
      renderer.autoClear = false; renderer.render(qScene, qCam); renderer.autoClear = auto;
      renderer.setScissorTest(false); renderer.setViewport(0, 0, size.x, size.y); renderer.setScissor(0, 0, size.x, size.y);
    },
    dispose() { rt.dispose(); mat.dispose(); quad.geometry.dispose(); },
  };
}
