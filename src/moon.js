// The Moon in the orbit's sky, where the ephemeris puts it (src/ephem.js; src/space.js places it): NASA's LRO colour
// mosaic with the LOLA relief as a normal map (textures/moon, CGI Moon Kit), lit by the Lommel–Seeliger law of regolith
// (no limb darkening, a crisp airless terminator) with the opposition surge near full, so the phase falls out of the
// Sun's true direction; the night side glows faintly with earthshine, strongest when the Earth seen from the Moon is
// full (a young crescent's "old Moon in the new Moon's arms"). Tidally locked: the near side faces the Earth.
import * as THREE from 'three';

export function createMoon({ texturePath = 'textures/' } = {}) {
  const load = (f, srgb) => { const t = new THREE.TextureLoader().load(texturePath + 'moon/' + f); if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 4; return t; };
  const mat = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: load('moon_color_2k.jpg', true) }, uNormal: { value: load('moon_normal_2k.jpg', false) }, uSun: { value: new THREE.Vector3(1, 0, 0) },
      uEarth: { value: new THREE.Vector3(0, -1, 0) }, uShine: { value: 0 }, uGain: { value: 1.3 } },
    vertexShader: /* glsl */`
      varying vec2 vUv; varying vec3 vN, vE, vNo, vW, vAx;
      #include <common>
      #include <logdepthbuf_pars_vertex>
      void main() {
        vUv = uv; vec3 n = normalize(position), e = normalize(cross(vec3(0.0, 1.0, 0.0), n + vec3(0.0, 0.0, 1e-4)));   // east, then north: the equirectangular map's frame
        mat3 m = mat3(modelMatrix); vAx = normalize(m * vec3(0.0, 1.0, 0.0)); vN = normalize(m * n); vE = normalize(m * e); vNo = normalize(m * cross(n, e));
        vec4 wp = modelMatrix * vec4(position, 1.0); vW = wp.xyz; gl_Position = projectionMatrix * viewMatrix * wp;
        #include <logdepthbuf_vertex>
      }`,
    fragmentShader: /* glsl */`
      uniform sampler2D uColor, uNormal; uniform vec3 uSun, uEarth; uniform float uShine, uGain; varying vec2 vUv; varying vec3 vN, vE, vNo, vW, vAx;
      #include <logdepthbuf_pars_fragment>
      void main() {
        #include <logdepthbuf_fragment>
        vec3 N = normalize(vN), c = cross(vAx, N); float l = length(c); vec3 E = l > 1e-4 ? c / l : normalize(vE), No = cross(N, E);   // east/north per pixel: no pinwheel at the poles
        vec3 t = texture2D(uNormal, vUv).xyz * 2.0 - 1.0; t.xy *= smoothstep(0.0, 0.02, l);
        vec3 n = normalize(t.x * E + t.y * No + t.z * N), v = normalize(cameraPosition - vW);
        vec3 alb = texture2D(uColor, vUv).rgb;
        float mu0 = max(dot(n, uSun), 0.0), mu = max(dot(n, v), 0.02), g = acos(clamp(dot(uSun, v), -1.0, 1.0));
        float ls = mu0 / (mu0 + mu) * (1.0 + 0.35 * exp(-g / 0.08));            // Lommel–Seeliger with the opposition surge
        float shade = smoothstep(-0.02, 0.06, dot(vN, uSun));                  // the geometric terminator stays crisp under the relief
        vec3 col = alb * uGain * ls * 2.0 * shade + alb * vec3(0.55, 0.65, 0.85) * uShine * max(dot(n, uEarth), 0.0);
        gl_FragColor = vec4(col, 1.0);
      }`,
  });
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 48), mat); mesh.name = 'moon'; mesh.renderOrder = -4;
  const _x = new THREE.Vector3(), _y = new THREE.Vector3(), _z = new THREE.Vector3(), _m = new THREE.Matrix4(); let hi = false, disposed = false;
  return {
    mesh, mat,
    // pos, radius: where to draw it (the true direction at a scaled distance, the true angular size); sun, toEarth, north: unit vectors
    place(pos, radius, sun, toEarth, north, earthPhase, toEarthWorld = toEarth) {   // toEarth/north in the parent's frame; sun and toEarthWorld in the world (the shader's)
      mesh.position.copy(pos); mesh.scale.setScalar(radius);
      _x.copy(toEarth); _y.copy(north).addScaledVector(_x, -north.dot(_x)).normalize(); _z.crossVectors(_x, _y);   // local +x (longitude 0) at the Earth, +y north
      mesh.quaternion.setFromRotationMatrix(_m.makeBasis(_x, _y, _z));
      mat.uniforms.uSun.value.copy(sun); mat.uniforms.uEarth.value.copy(toEarthWorld); mat.uniforms.uShine.value = 0.004 * earthPhase;
    },
    hiRes() {                                                       // lunar orbit: the 4K maps (≈ 4.5 MB), fetched once and swapped in together when both arrive
      if (hi) return; hi = true; const got = {};
      for (const [u, f, srgb] of [['uColor', 'moon_color_4k.jpg', true], ['uNormal', 'moon_normal_4k.jpg', false]]) new THREE.TextureLoader().load(texturePath + 'moon/' + f, (t) => {
        if (srgb) t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; got[u] = t; if (Object.keys(got).length < 2) return;
        for (const k in got) { if (disposed) got[k].dispose(); else { mat.uniforms[k].value.dispose(); mat.uniforms[k].value = got[k]; } }
      }, undefined, () => console.warn('moon: ' + f + ' failed to load, keeping the 2K maps'));
    },
    dispose() { disposed = true; mesh.geometry.dispose(); mat.uniforms.uColor.value.dispose(); mat.uniforms.uNormal.value.dispose(); mat.dispose(); },
  };
}
