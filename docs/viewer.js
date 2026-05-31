/* SHA-256 circuit explorer — vanilla WebGL2 viewer for .glayout files.
 *
 * Design (see plan.md / OPTIMIZATIONS.md for the circuit side):
 *   - The .glayout preprocessor (layout-graph.py) reindexes nodes in topological
 *     order and precomputes 2-D positions. So here: simulation is a single forward
 *     loop over typed arrays, and rendering is two instanced/batched GPU draws.
 *   - Nodes: one instanced unit quad per node; a fragment shader draws the gate
 *     symbol (NAND = flat-top/round-bottom D + output bubble) with a 0/1/X digit
 *     (glyph atlas) at high zoom, a dot at low zoom (LOD). State (0/1/X) is read in
 *     the shader from a GPU "state texture" indexed by gl_InstanceID.
 *   - Wires: every edge is two GL_LINES vertices; color = driver's state (sampled
 *     from the same state texture). Faded out when zoomed far (LOD).
 *   - Light-cone: hovering picks the nearest node (spatial grid), BFS up (fanins)
 *     and down (fanouts) marks a "highlight texture"; the shaders dim everything
 *     outside the cone.
 *
 * Everything that changes per-frame from a pan/zoom is just two uniforms, so it
 * stays at 60fps on a ~230k-node / ~460k-edge graph.
 */
"use strict";

// ---- node type codes (must match layout-graph.py) -------------------------
const T_INPUT = 0, T_NAND = 1, T_OUTPUT = 2, T_CONST0 = 3, T_CONST1 = 4;

// ---- world geometry constants ---------------------------------------------
const NODE_HALF = [0.42, 0.58];   // gate half-size in world units
const PORT_DX = 0.22;             // x offset of a NAND's two input ports
const MIN_NODE_PX = 1.2;          // gates never shrink below this (overview dots)

// ---- DOM ------------------------------------------------------------------
const canvas = document.getElementById("gl");
const elMsg = document.getElementById("msg");
const elHash = document.getElementById("hash");
const elHashOk = document.getElementById("hashok");
const elStats = document.getElementById("stats");
const elHover = document.getElementById("hover");
const elDrop = document.getElementById("drop");
const elFile = document.getElementById("file");
const elToast = document.getElementById("toast");

let gl;
let DPR = 1;

// ---- graph data (filled by load) ------------------------------------------
let meta = null;
let N = 0;
let posX, posY, type, fanin0, fanin1, bitIdx, inByBit, outByBit;
let posXY;                 // interleaved [x0,y0,x1,y1,...] for the GPU
let state, hilite, seen2;  // Uint8Array(N)
let inputVal;              // Uint8Array(#input bits) 0/1
let outStart, outList;     // fanout CSR for light-cone descend
let bounds;
// spatial pick grid
const CELL = 3.0;
let grid, gridCols;

// ---- GL objects -----------------------------------------------------------
let progNode, progWire;
let vaoNode, vaoWire;
let stateTex, hiliteTex, glyphTex, texW, texH;
let wireVertexCount = 0;
let uNode = {}, uWire = {};

// ---- camera ---------------------------------------------------------------
const cam = { x: 0, y: 0, zoom: 1 };
let minZoom = 0.01, maxZoom = 160;
let wiresOn = true;
let hoverActive = false, lockedNode = -1, pickedNode = -1;

let needDraw = false;
function requestDraw() { if (!needDraw) { needDraw = true; requestAnimationFrame(render); } }
function toast(msg, ms = 2200) { elToast.textContent = msg; elToast.classList.add("show"); clearTimeout(toast._t); toast._t = setTimeout(() => elToast.classList.remove("show"), ms); }

// ===========================================================================
// Shaders
// ===========================================================================
const NODE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aQuad;
layout(location=1) in vec2 aPos;
layout(location=2) in uint aType;
uniform vec2 uCam; uniform float uZoom; uniform vec2 uViewport; uniform vec2 uNodeHalf;
uniform highp usampler2D uStateTex; uniform highp usampler2D uHiliteTex; uniform int uTexW;
out vec2 vLocal; out float vState; flat out uint vType; flat out uint vHi; out float vPx;
void main(){
  int id = gl_InstanceID;
  ivec2 tc = ivec2(id % uTexW, id / uTexW);
  vState = float(texelFetch(uStateTex, tc, 0).r);
  vHi = texelFetch(uHiliteTex, tc, 0).r;
  vType = aType; vLocal = aQuad;
  vec2 centerScreen = (aPos - uCam) * uZoom + uViewport * 0.5;
  vec2 halfPx = max(uNodeHalf * uZoom, vec2(${MIN_NODE_PX.toFixed(1)}));
  vec2 screen = centerScreen + aQuad * halfPx;
  gl_Position = vec4(screen.x / uViewport.x * 2.0 - 1.0, 1.0 - screen.y / uViewport.y * 2.0, 0.0, 1.0);
  vPx = halfPx.x * 2.0;
}`;

const NODE_FS = `#version 300 es
precision highp float;
in vec2 vLocal; in float vState; flat in uint vType; flat in uint vHi; in float vPx;
uniform sampler2D uGlyph; uniform int uHoverActive;
out vec4 frag;
const vec3 C_ONE = vec3(0.247,0.725,0.314);
const vec3 C_ZERO= vec3(0.294,0.416,0.588);
const vec3 C_X   = vec3(0.823,0.600,0.133);
float sdBox(vec2 p, vec2 b){ vec2 d=abs(p)-b; return length(max(d,0.0))+min(max(d.x,d.y),0.0); }
float sdCircle(vec2 p, float r){ return length(p)-r; }
void main(){
  vec3 col = vState>1.5 ? C_X : (vState>0.5 ? C_ONE : C_ZERO);
  float aa = fwidth(vLocal.x) * 1.5 + 0.002;
  float d;
  if (vType == 1u) {                       // NAND D-symbol, flat top, round bottom + bubble
    float dBody = min( sdBox(vLocal - vec2(0.0,-0.30), vec2(0.70,0.42)),
                       sdCircle(vLocal - vec2(0.0, 0.12), 0.70) );
    float dBub  = sdCircle(vLocal - vec2(0.0, 0.92), 0.13);
    d = min(dBody, dBub);
  } else {                                  // input / output / const: rounded square
    d = sdBox(vLocal, vec2(0.74)) - 0.14;
  }
  float mask = 1.0 - smoothstep(0.0, aa, d);
  if (mask < 0.004) discard;
  float alpha = mask;

  if (vPx >= 15.0) {                         // detail: digit + edge shading
    vec2 g = vLocal * 0.5 + 0.5;
    float cell = vState>1.5 ? 2.0 : (vState>0.5 ? 1.0 : 0.0);
    float glyph = texture(uGlyph, vec2((g.x + cell)/3.0, g.y)).a;
    col = mix(col, vec3(0.96), glyph * 0.92);
    float edge = 1.0 - smoothstep(aa*1.0, aa*3.0, abs(d));
    col = mix(col, col*0.45, edge*0.7);
  }

  if (uHoverActive == 1) {
    if (vHi == 0u) { col = mix(col, vec3(0.10,0.12,0.16), 0.82); alpha *= 0.30; }
    else if (vHi == 2u) { col = mix(col, vec3(1.0), 0.40); }
  }
  frag = vec4(col, alpha);
}`;

const WIRE_VS = `#version 300 es
precision highp float;
layout(location=0) in vec2 aWPos;
layout(location=1) in uint aDriver;
layout(location=2) in uint aTarget;
uniform vec2 uCam; uniform float uZoom; uniform vec2 uViewport;
uniform highp usampler2D uStateTex; uniform highp usampler2D uHiliteTex; uniform int uTexW;
uniform int uHoverActive; uniform float uWireAlpha;
out vec4 vColor;
const vec3 C_ONE = vec3(0.247,0.725,0.314);
const vec3 C_ZERO= vec3(0.294,0.416,0.588);
const vec3 C_X   = vec3(0.823,0.600,0.133);
void main(){
  ivec2 dc = ivec2(int(aDriver) % uTexW, int(aDriver) / uTexW);
  float s = float(texelFetch(uStateTex, dc, 0).r);
  vec3 c = s>1.5 ? C_X : (s>0.5 ? C_ONE : C_ZERO);
  float a = uWireAlpha;
  if (uHoverActive == 1) {
    uint hd = texelFetch(uHiliteTex, dc, 0).r;
    ivec2 tc = ivec2(int(aTarget) % uTexW, int(aTarget) / uTexW);
    uint ht = texelFetch(uHiliteTex, tc, 0).r;
    if (hd != 0u && ht != 0u) { a = min(1.0, a*1.5 + 0.55); c = mix(c, vec3(1.0), 0.12); }
    else { a *= 0.08; }
  }
  vColor = vec4(c, a);
  vec2 screen = (aWPos - uCam) * uZoom + uViewport * 0.5;
  gl_Position = vec4(screen.x / uViewport.x * 2.0 - 1.0, 1.0 - screen.y / uViewport.y * 2.0, 0.0, 1.0);
}`;

const WIRE_FS = `#version 300 es
precision highp float;
in vec4 vColor; out vec4 frag;
void main(){ if (vColor.a < 0.01) discard; frag = vColor; }`;

// ===========================================================================
// GL helpers
// ===========================================================================
function compile(src, kind) {
  const s = gl.createShader(kind);
  gl.shaderSource(s, src); gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + "\n" + src);
  return s;
}
function link(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(vs, gl.VERTEX_SHADER));
  gl.attachShader(p, compile(fs, gl.FRAGMENT_SHADER));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}
function uniforms(p, names) { const u = {}; for (const n of names) u[n] = gl.getUniformLocation(p, n); return u; }

// ===========================================================================
// Load + parse .glayout
// ===========================================================================
function parseGlayout(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "GLAY") throw new Error("not a .glayout file (bad magic)");
  const headerLen = dv.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 16, headerLen)));
  let ds = 16 + headerLen; ds += (4 - (ds % 4)) % 4;
  const mk = { float32: Float32Array, int32: Int32Array, uint8: Uint8Array };
  const arr = {};
  for (const a of header.arrays) arr[a.name] = new mk[a.dtype](buf, ds + a.offset, a.count);
  return { header, arr };
}

function loadData(buf) {
  const { header, arr } = parseGlayout(buf);
  meta = header; N = header.nodeCount; bounds = header.bounds;
  posX = arr.posX; posY = arr.posY; type = arr.type;
  fanin0 = arr.fanin0; fanin1 = arr.fanin1; bitIdx = arr.bit;
  inByBit = arr.inputIdxByBit; outByBit = arr.outputIdxByBit;

  posXY = new Float32Array(N * 2);
  for (let i = 0; i < N; i++) { posXY[2*i] = posX[i]; posXY[2*i+1] = posY[i]; }
  state = new Uint8Array(N); hilite = new Uint8Array(N); seen2 = new Uint8Array(N);
  inputVal = new Uint8Array(inByBit.length);

  buildFanout();
  buildGrid();
  initGL();
  buildWires();

  // initial readable view: top-center, gates ~22px
  cam.zoom = 26; cam.x = 0; cam.y = (NODE_HALF[1] + 1) * meta.dy * 3.0;
  const fitW = canvas.width / ((bounds.maxX - bounds.minX + 4) || 1);
  const fitAll = Math.min(fitW, canvas.height / ((bounds.maxY - bounds.minY + 4) || 1));
  minZoom = Math.min(fitAll * 0.6, 0.02);

  elDrop.classList.remove("show");
  elStats.innerHTML =
    `<b>${N.toLocaleString()}</b> nodes · <b>${(wireVertexCount/2).toLocaleString()}</b> wires<br>` +
    `${meta.inputCount} inputs · ${meta.outputCount} outputs · ${meta.layerCount.toLocaleString()} layers deep`;
  setMessage(elMsg.value);
  requestDraw();
}

function buildFanout() {
  const deg = new Int32Array(N);
  for (let i = 0; i < N; i++) { const a = fanin0[i], b = fanin1[i]; if (a >= 0) deg[a]++; if (b >= 0) deg[b]++; }
  outStart = new Int32Array(N + 1);
  for (let i = 0; i < N; i++) outStart[i + 1] = outStart[i] + deg[i];
  outList = new Int32Array(outStart[N]);
  const cur = outStart.slice();
  for (let i = 0; i < N; i++) { const a = fanin0[i], b = fanin1[i]; if (a >= 0) outList[cur[a]++] = i; if (b >= 0) outList[cur[b]++] = i; }
}

function buildGrid() {
  gridCols = Math.ceil((bounds.maxX - bounds.minX) / CELL) + 2;
  grid = new Map();
  for (let i = 0; i < N; i++) {
    const cx = Math.floor((posX[i] - bounds.minX) / CELL);
    const cy = Math.floor((posY[i] - bounds.minY) / CELL);
    const k = cy * gridCols + cx;
    let a = grid.get(k); if (!a) { a = []; grid.set(k, a); } a.push(i);
  }
}

function pickAt(wx, wy, radius) {
  const cx = Math.floor((wx - bounds.minX) / CELL);
  const cy = Math.floor((wy - bounds.minY) / CELL);
  let best = -1, bd = radius * radius;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    const a = grid.get((cy + dy) * gridCols + (cx + dx)); if (!a) continue;
    for (const i of a) { const ex = posX[i]-wx, ey = posY[i]-wy, dd = ex*ex+ey*ey; if (dd < bd) { bd = dd; best = i; } }
  }
  return best;
}

// ===========================================================================
// GL init: programs, geometry, textures
// ===========================================================================
function initGL() {
  if (progNode) return;            // once
  progNode = link(NODE_VS, NODE_FS);
  progWire = link(WIRE_VS, WIRE_FS);
  uNode = uniforms(progNode, ["uCam","uZoom","uViewport","uNodeHalf","uStateTex","uHiliteTex","uTexW","uGlyph","uHoverActive"]);
  uWire = uniforms(progWire, ["uCam","uZoom","uViewport","uStateTex","uHiliteTex","uTexW","uHoverActive","uWireAlpha"]);

  // node quad (two triangles) + per-instance pos/type
  vaoNode = gl.createVertexArray(); gl.bindVertexArray(vaoNode);
  const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
  const qb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, qb); gl.bufferData(gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, posXY, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0); gl.vertexAttribDivisor(1, 1);
  const tb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, tb); gl.bufferData(gl.ARRAY_BUFFER, type, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(2); gl.vertexAttribIPointer(2, 1, gl.UNSIGNED_BYTE, 0, 0); gl.vertexAttribDivisor(2, 1);
  gl.bindVertexArray(null);

  // state + highlight textures (R8UI, indexed by node id)
  texW = Math.ceil(Math.sqrt(N)); texH = Math.ceil(N / texW);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  stateTex = makeU8Tex(); hiliteTex = makeU8Tex();
  glyphTex = makeGlyphAtlas();
}

function makeU8Tex() {
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8UI, texW, texH);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}
function uploadU8(tex, data) {            // data: Uint8Array(N)
  const buf = new Uint8Array(texW * texH); buf.set(data);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, texW, texH, gl.RED_INTEGER, gl.UNSIGNED_BYTE, buf);
}

function makeGlyphAtlas() {
  const cell = 64, cv = document.createElement("canvas"); cv.width = cell*3; cv.height = cell;
  const c = cv.getContext("2d");
  c.clearRect(0,0,cv.width,cv.height);
  c.fillStyle = "#fff"; c.textAlign = "center"; c.textBaseline = "middle";
  c.font = "bold 46px ui-monospace, Consolas, monospace";
  ["0","1","X"].forEach((ch,i) => c.fillText(ch, i*cell + cell/2, cell/2 + 2));
  const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cv);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function buildWires() {
  // one line segment per edge: driver output (bottom) -> target input port (top)
  let E = 0;
  for (let i = 0; i < N; i++) { if (fanin0[i] >= 0) E++; if (fanin1[i] >= 0) E++; }
  const pos = new Float32Array(E * 4);
  const drv = new Uint32Array(E * 2);
  const tgt = new Uint32Array(E * 2);
  let v = 0;   // vertex index
  const addEdge = (driver, target, slot) => {
    const sx = posX[driver], sy = posY[driver] + NODE_HALF[1];           // driver bottom
    const ex = posX[target] + slot * PORT_DX, ey = posY[target] - NODE_HALF[1]; // target top
    pos[v*2] = sx; pos[v*2+1] = sy; drv[v] = driver; tgt[v] = target; v++;
    pos[v*2] = ex; pos[v*2+1] = ey; drv[v] = driver; tgt[v] = target; v++;
  };
  for (let i = 0; i < N; i++) {
    const a = fanin0[i], b = fanin1[i];
    if (a >= 0 && b >= 0) { addEdge(a, i, -1); addEdge(b, i, +1); }     // NAND: two ports
    else if (a >= 0) { addEdge(a, i, 0); }                              // output: one port
  }
  wireVertexCount = v;

  vaoWire = gl.createVertexArray(); gl.bindVertexArray(vaoWire);
  const pb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const db = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, db); gl.bufferData(gl.ARRAY_BUFFER, drv, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribIPointer(1, 1, gl.UNSIGNED_INT, 0, 0);
  const gb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, gb); gl.bufferData(gl.ARRAY_BUFFER, tgt, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(2); gl.vertexAttribIPointer(2, 1, gl.UNSIGNED_INT, 0, 0);
  gl.bindVertexArray(null);
}

// ===========================================================================
// Simulation
// ===========================================================================
function simulate() {
  for (let i = 0; i < N; i++) {
    const t = type[i];
    if (t === T_NAND) state[i] = 1 - (state[fanin0[i]] & state[fanin1[i]]);
    else if (t === T_INPUT) state[i] = inputVal[bitIdx[i]];
    else if (t === T_OUTPUT) state[i] = state[fanin0[i]];
    else if (t === T_CONST1) state[i] = 1;
    else state[i] = 0;
  }
}

function setMessage(str) {
  let bytes = new TextEncoder().encode(str);
  if (bytes.length > 55) { bytes = bytes.slice(0, 55); toast("clamped to 55 bytes (single block)"); }
  const block = new Uint8Array(64);
  block.set(bytes); block[bytes.length] = 0x80;
  const bitlen = bytes.length * 8;                       // < 2^32, high word stays 0
  block[60] = (bitlen >>> 24) & 255; block[61] = (bitlen >>> 16) & 255;
  block[62] = (bitlen >>> 8) & 255; block[63] = bitlen & 255;
  for (let byteI = 0; byteI < 64; byteI++)
    for (let k = 0; k < 8; k++) inputVal[byteI*8 + k] = (block[byteI] >> (7 - k)) & 1;  // k=0 -> MSB
  simulate();
  uploadU8(stateTex, state);
  updateHash(bytes);
  requestDraw();
}

function updateHash(bytes) {
  let out = "", acc = 0, cnt = 0;
  for (let hb = 0; hb < 256; hb++) {                     // HASH-000 = MSB
    acc = (acc << 1) | state[outByBit[hb]]; cnt++;
    if (cnt === 4) { out += acc.toString(16); acc = 0; cnt = 0; }
  }
  elHash.textContent = out;
  const ref = (typeof sha256Hex === "function") ? sha256Hex(bytes) : null;
  if (ref === null) { elHashOk.textContent = ""; }
  else if (ref === out) { elHashOk.innerHTML = `<span class="ok">✓ matches SHA-256</span>`; }
  else { elHashOk.innerHTML = `<span class="bad">✗ mismatch</span> (ref ${ref.slice(0,12)}…)`; }
}

// ===========================================================================
// Light-cone (ancestors + descendants of a node)
// ===========================================================================
function markCone(start) {
  hilite.fill(0);
  let up = 0, down = 0;
  let stack = [start];
  while (stack.length) {                                 // up: fanins
    const u = stack.pop();
    if (hilite[u]) continue;
    hilite[u] = 1; up++;
    const a = fanin0[u], b = fanin1[u];
    if (a >= 0) stack.push(a); if (b >= 0) stack.push(b);
  }
  seen2.fill(0); stack = [start];
  while (stack.length) {                                 // down: fanouts
    const u = stack.pop();
    if (seen2[u]) continue;
    seen2[u] = 1; hilite[u] = 1; down++;
    for (let e = outStart[u]; e < outStart[u + 1]; e++) stack.push(outList[e]);
  }
  hilite[start] = 2;
  uploadU8(hiliteTex, hilite);
  return { up: up - 1, down: down - 1 };                 // exclude the node itself
}

function nodeLabel(i) {
  const t = type[i];
  if (t === T_INPUT) return `MESSAGE-${String(bitIdx[i]).padStart(3,"0")}`;
  if (t === T_OUTPUT) return `HASH-${String(bitIdx[i]).padStart(3,"0")}`;
  if (t === T_CONST0 || t === T_CONST1) return `const ${t === T_CONST1 ? 1 : 0}`;
  return `NAND #${i}`;
}

function setHover(node) {
  if (node === pickedNode) return;
  pickedNode = node;
  if (node < 0) {
    hoverActive = false;
    elHover.innerHTML = `<span class="hint">hover a gate or wire to trace its light-cone</span>`;
  } else {
    const c = markCone(node);
    hoverActive = true;
    const s = state[node], sl = s > 1 ? "X" : s;
    const depth = Math.round(posY[node] / meta.dy);
    elHover.innerHTML =
      `<span class="node-id">${nodeLabel(node)}</span> · state <b>${sl}</b> · depth ${depth}<br>` +
      `<span class="hint">light-cone: ↑ ${c.up.toLocaleString()} feed-in · ↓ ${c.down.toLocaleString()} dependent</span>`;
  }
  requestDraw();
}

// ===========================================================================
// Render
// ===========================================================================
function render() {
  needDraw = false;
  if (!gl || !meta) return;
  gl.clearColor(0.043, 0.055, 0.078, 1.0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  const VP = [canvas.width, canvas.height];
  const wireAlpha = wiresOn ? 0.62 * smoothstep(4.0, 12.0, cam.zoom) : 0.0;

  // wires first (behind nodes)
  if (wireAlpha > 0.001 || hoverActive) {
    gl.useProgram(progWire); gl.bindVertexArray(vaoWire);
    gl.uniform2f(uWire.uCam, cam.x, cam.y); gl.uniform1f(uWire.uZoom, cam.zoom);
    gl.uniform2f(uWire.uViewport, VP[0], VP[1]); gl.uniform1i(uWire.uTexW, texW);
    gl.uniform1i(uWire.uHoverActive, hoverActive ? 1 : 0);
    gl.uniform1f(uWire.uWireAlpha, Math.max(wireAlpha, hoverActive ? 0.0 : 0.0));
    bindTex(uWire, 0, 1);
    gl.drawArrays(gl.LINES, 0, wireVertexCount);
  }

  // nodes
  gl.useProgram(progNode); gl.bindVertexArray(vaoNode);
  gl.uniform2f(uNode.uCam, cam.x, cam.y); gl.uniform1f(uNode.uZoom, cam.zoom);
  gl.uniform2f(uNode.uViewport, VP[0], VP[1]); gl.uniform2f(uNode.uNodeHalf, NODE_HALF[0], NODE_HALF[1]);
  gl.uniform1i(uNode.uTexW, texW); gl.uniform1i(uNode.uHoverActive, hoverActive ? 1 : 0);
  bindTex(uNode, 0, 1);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, glyphTex); gl.uniform1i(uNode.uGlyph, 2);
  gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, N);

  gl.bindVertexArray(null);
}
function bindTex(u, stateUnit, hiliteUnit) {
  gl.activeTexture(gl.TEXTURE0 + stateUnit); gl.bindTexture(gl.TEXTURE_2D, stateTex); gl.uniform1i(u.uStateTex, stateUnit);
  gl.activeTexture(gl.TEXTURE0 + hiliteUnit); gl.bindTexture(gl.TEXTURE_2D, hiliteTex); gl.uniform1i(u.uHiliteTex, hiliteUnit);
}
function smoothstep(a, b, x) { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); }

// ===========================================================================
// Camera / interaction
// ===========================================================================
function screenToWorld(sx, sy) { return { x: (sx - canvas.width/2)/cam.zoom + cam.x, y: (sy - canvas.height/2)/cam.zoom + cam.y }; }

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.floor(canvas.clientWidth * DPR);
  canvas.height = Math.floor(canvas.clientHeight * DPR);
  if (gl) gl.viewport(0, 0, canvas.width, canvas.height);
  requestDraw();
}
function fitWidth() {
  if (!meta) return;
  const w = (bounds.maxX - bounds.minX) || 1;
  cam.zoom = Math.max(minZoom, canvas.width / (w * 1.08));
  cam.x = (bounds.minX + bounds.maxX) / 2; cam.y = (bounds.minY + bounds.maxY) / 2;
  requestDraw();
}

// Unified pointer interaction (mouse + touch + pen): one-pointer pan, two-pointer
// pinch-zoom+pan anchored to the finger midpoint, tap/click to lock the cone, and
// mouse-only hover. touch-action:none on the canvas keeps the browser from hijacking.
const pointers = new Map();              // active pointerId -> {x,y} in device px
let panId = -1, panLast = null, downStart = null, movedFar = false, pinchPrev = null;
let hoverRAF = 0, pendingHover = null;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function devXY(e) { const r = canvas.getBoundingClientRect(); return { x: (e.clientX - r.left) * DPR, y: (e.clientY - r.top) * DPR }; }
function pinchState(p) { const dx = p[0].x - p[1].x, dy = p[0].y - p[1].y; return { dist: Math.hypot(dx, dy) || 1, mx: (p[0].x + p[1].x) / 2, my: (p[0].y + p[1].y) / 2 }; }
function tapPick(pos) {
  if (!meta || !pos) return;
  const w = screenToWorld(pos.x, pos.y);
  const n = pickAt(w.x, w.y, Math.min(CELL, 18 / cam.zoom));
  if (n >= 0) { lockedNode = (lockedNode === n) ? -1 : n; setHover(lockedNode >= 0 ? lockedNode : -1); }
  else { lockedNode = -1; setHover(-1); }
}

canvas.addEventListener("pointerdown", e => {
  e.preventDefault();
  try { canvas.setPointerCapture(e.pointerId); } catch (_) {}
  const p = devXY(e); pointers.set(e.pointerId, p);
  if (pointers.size === 1) { panId = e.pointerId; panLast = p; downStart = p; movedFar = false; pinchPrev = null; }
  else if (pointers.size === 2) { panId = -1; movedFar = true; pinchPrev = pinchState([...pointers.values()]); }
}, { passive: false });

canvas.addEventListener("pointermove", e => {
  if (!pointers.has(e.pointerId)) {                      // un-pressed mouse move -> hover light-cone
    if (e.pointerType === "mouse" && meta && lockedNode < 0) {
      pendingHover = devXY(e);
      if (!hoverRAF) hoverRAF = requestAnimationFrame(() => {
        hoverRAF = 0; const w = screenToWorld(pendingHover.x, pendingHover.y);
        setHover(pickAt(w.x, w.y, Math.min(CELL, 18 / cam.zoom)));
      });
    }
    return;
  }
  e.preventDefault();
  const p = devXY(e); pointers.set(e.pointerId, p);
  if (pointers.size >= 2) {                               // pinch: zoom + pan anchored to midpoint
    const cur = pinchState([...pointers.values()]);
    if (pinchPrev) {
      const before = screenToWorld(pinchPrev.mx, pinchPrev.my);
      cam.zoom = clamp(cam.zoom * (cur.dist / pinchPrev.dist), minZoom, maxZoom);
      const after = screenToWorld(cur.mx, cur.my);
      cam.x += before.x - after.x; cam.y += before.y - after.y;
      requestDraw();
    }
    pinchPrev = cur; return;
  }
  if (e.pointerId === panId && panLast) {                 // single-pointer pan
    cam.x -= (p.x - panLast.x) / cam.zoom;
    cam.y -= (p.y - panLast.y) / cam.zoom;
    if (downStart && Math.hypot(p.x - downStart.x, p.y - downStart.y) > 6 * DPR) movedFar = true;
    panLast = p; requestDraw();
  }
}, { passive: false });

function endPointer(e) {
  try { canvas.releasePointerCapture(e.pointerId); } catch (_) {}
  const wasSize = pointers.size, wasPan = e.pointerId === panId;
  pointers.delete(e.pointerId);
  if (wasSize === 1 && wasPan && !movedFar) tapPick(downStart);   // tap / click = lock cone
  if (pointers.size === 1) {                              // 2->1: rebase pan to remaining finger
    const [id, pos] = [...pointers.entries()][0];
    panId = id; panLast = pos; downStart = pos; movedFar = true; pinchPrev = null;
  } else if (pointers.size === 0) { panId = -1; panLast = null; pinchPrev = null; }
}
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);

canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const s = devXY(e);
  const before = screenToWorld(s.x, s.y);
  cam.zoom = clamp(cam.zoom * Math.exp(-e.deltaY * 0.0015), minZoom, maxZoom);
  const after = screenToWorld(s.x, s.y);
  cam.x += before.x - after.x; cam.y += before.y - after.y;
  requestDraw();
}, { passive: false });

window.addEventListener("keydown", e => {
  if (e.target === elMsg) return;
  if (e.key === "f") fitWidth();
  else if (e.key === "w") { wiresOn = !wiresOn; toast("wires " + (wiresOn ? "on" : "off")); requestDraw(); }
  else if (e.key === "Escape") { lockedNode = -1; setHover(-1); }
});
elMsg.addEventListener("input", () => setMessage(elMsg.value));

// ===========================================================================
// File loading (fetch when served; drag/drop or picker otherwise)
// ===========================================================================
function readFile(file) {
  const fr = new FileReader();
  fr.onload = () => { try { loadData(fr.result); } catch (err) { toast("load failed: " + err.message, 5000); console.error(err); } };
  fr.readAsArrayBuffer(file);
}
elFile.addEventListener("change", e => { if (e.target.files[0]) readFile(e.target.files[0]); });
["dragover","drop"].forEach(t => window.addEventListener(t, e => e.preventDefault()));
window.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) readFile(f); });

async function tryFetch() {
  for (const url of ["SHA256-opt.glayout", "../SHA256-opt.glayout", "SHA256-base.glayout", "../SHA256-base.glayout"]) {
    try {
      const r = await fetch(url); if (!r.ok) continue;
      loadData(await r.arrayBuffer()); toast("loaded " + url); return true;
    } catch (_) {}
  }
  return false;
}

// ===========================================================================
// Boot
// ===========================================================================
// debug/inspection handle (also handy for future iteration & scripted tours)
window.DBG = {
  get cam() { return cam; },
  info() { return { N, zoom: cam.zoom, camX: cam.x, camY: cam.y, bounds, layers: meta && meta.layerCount, wires: wireVertexCount / 2 }; },
  node(i) { return { type: type[i], state: state[i], x: posX[i], y: posY[i], bit: bitIdx[i], depth: Math.round(posY[i] / meta.dy) }; },
  goto(z, wx, wy) { cam.zoom = z; if (wx !== undefined) { cam.x = wx; cam.y = wy; } requestDraw(); },
  fit: fitWidth,
  hoverNode(i) { lockedNode = i; setHover(i); },
  clearHover() { lockedNode = -1; setHover(-1); },
  pick: (wx, wy, r) => pickAt(wx, wy, r === undefined ? 16 / cam.zoom : r),
};

function boot() {
  gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
  if (!gl) { document.body.innerHTML = "<p style='padding:40px'>WebGL2 is required and not available in this browser.</p>"; return; }
  resize(); window.addEventListener("resize", resize);
  tryFetch().then(ok => { if (!ok) { elDrop.classList.add("show"); elStats.textContent = "drop a .glayout file to begin"; } });
}
boot();
