// Voice Chat Local — browser client.
// Captures mic audio (hold-to-talk), streams 16 kHz PCM over a WebSocket, and
// plays streamed reply sentences back gaplessly and in order using a clock-
// scheduled FIFO on the Web Audio API (research.md D8; FR-010, SC-006).

const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const aiCaptionEl = document.getElementById("ai-caption");
const talkBtn = document.getElementById("talk-btn");
const langButtons = document.querySelectorAll(".lang-btn");

let currentLang = "en"; // reply language chosen in the UI
let ws = null;
let captureCtx = null; // AudioContext @16k for mic capture
let playbackCtx = null; // AudioContext for reply playback
let mediaStream = null;
let workletNode = null;
let sourceNode = null;
let isRecording = false;

let playbackSampleRate = 22050;
let nextStartTime = 0; // scheduling cursor for gapless playback

function setStatus(text) {
  statusEl.textContent = text;
}

// ---- WebSocket ------------------------------------------------------------
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/voice`);
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    setStatus("Conectado");
    sendLanguage(currentLang); // tell the backend the current choice
  };
  ws.onclose = () => {
    setStatus("Desconectado");
    talkBtn.disabled = true;
  };
  ws.onerror = () => setStatus("Erro de conexão");

  ws.onmessage = (event) => {
    if (typeof event.data === "string") {
      handleControl(JSON.parse(event.data));
    } else {
      enqueueAudio(event.data); // ArrayBuffer of int16 PCM
    }
  };
}

function handleControl(msg) {
  switch (msg.type) {
    case "ready":
      talkBtn.disabled = false;
      setStatus("Pronto — segure para falar");
      break;
    case "transcript":
      transcriptEl.textContent = `você: ${msg.text}`;
      break;
    case "speaking":
      playbackSampleRate = msg.sample_rate || 22050;
      ensurePlaybackContext();
      nextStartTime = playbackCtx.currentTime;
      aiCaptionEl.textContent = ""; // start a fresh caption for this reply
      setStatus("Falando…");
      break;
    case "segment":
      // Build the live caption of what the AI is saying, sentence by sentence.
      aiCaptionEl.textContent = `${aiCaptionEl.textContent} ${msg.text}`.trim();
      break;
    case "no_speech":
      setStatus("Não entendi — tente de novo");
      break;
    case "reply_end":
      setStatus("Pronto — segure para falar");
      break;
    case "error":
      setStatus(`Erro: ${msg.message}`);
      stopPlayback();
      break;
  }
}

// ---- Language toggle ------------------------------------------------------
function sendLanguage(lang) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "set_language", value: lang }));
  }
}

function setLanguage(lang) {
  currentLang = lang;
  langButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });
  sendLanguage(lang);
}

langButtons.forEach((btn) => {
  btn.addEventListener("click", () => setLanguage(btn.dataset.lang));
});

// ---- Microphone capture ---------------------------------------------------
async function startMic() {
  if (isRecording || !ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus("Sem acesso ao microfone");
    return;
  }
  captureCtx = new AudioContext({ sampleRate: 16000 });
  await captureCtx.audioWorklet.addModule("/static/pcm-worklet.js");
  sourceNode = captureCtx.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(captureCtx, "pcm-processor");
  workletNode.port.onmessage = (e) => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(e.data);
  };
  sourceNode.connect(workletNode);
  // Do not connect to destination (avoid echo).

  isRecording = true;
  talkBtn.classList.add("recording");
  transcriptEl.textContent = "";
  aiCaptionEl.textContent = ""; // clear previous captions for the new turn
  setStatus("Ouvindo…");
  ws.send(JSON.stringify({ type: "talk_start" }));
}

function stopMic() {
  if (!isRecording) return;
  isRecording = false;
  talkBtn.classList.remove("recording");
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "talk_end" }));
  }
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
  if (workletNode) workletNode.disconnect();
  if (sourceNode) sourceNode.disconnect();
  if (captureCtx) captureCtx.close();
  captureCtx = null;
  workletNode = null;
  sourceNode = null;
  setStatus("Processando…");
}

// ---- Playback (gapless, ordered FIFO) -------------------------------------
function ensurePlaybackContext() {
  if (!playbackCtx || playbackCtx.state === "closed") {
    playbackCtx = new AudioContext();
  }
  if (playbackCtx.state === "suspended") playbackCtx.resume();
}

function enqueueAudio(arrayBuffer) {
  ensurePlaybackContext();
  const int16 = new Int16Array(arrayBuffer);
  if (int16.length === 0) return;
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

  const buffer = playbackCtx.createBuffer(1, float32.length, playbackSampleRate);
  buffer.copyToChannel(float32, 0);

  const node = playbackCtx.createBufferSource();
  node.buffer = buffer;
  node.connect(playbackCtx.destination);

  const now = playbackCtx.currentTime;
  if (nextStartTime < now) nextStartTime = now;
  node.start(nextStartTime);
  nextStartTime += buffer.duration; // schedule next segment right after this one
}

function stopPlayback() {
  if (playbackCtx) {
    playbackCtx.close();
    playbackCtx = null;
  }
  nextStartTime = 0;
}

// ---- Wire up controls -----------------------------------------------------
talkBtn.addEventListener("mousedown", startMic);
talkBtn.addEventListener("mouseup", stopMic);
talkBtn.addEventListener("mouseleave", stopMic);
talkBtn.addEventListener("touchstart", (e) => {
  e.preventDefault();
  startMic();
});
talkBtn.addEventListener("touchend", (e) => {
  e.preventDefault();
  stopMic();
});

connect();
