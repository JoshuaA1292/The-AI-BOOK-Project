// Ask the Orb — Hybrid client-only: WebLLM (WebGPU) + Transformers.js (WASM) fallback

// ---- DOM ----
const chat         = document.getElementById("chat");
const form         = document.getElementById("chatForm");
const input        = document.getElementById("chatInput");
const mystery      = document.getElementById("mysteryMode");
const fairnessForm = document.getElementById("fairnessForm");
const fairnessInput= document.getElementById("fairnessInput");
const creativeForm = document.getElementById("creativeForm");
const creativeInput= document.getElementById("creativeInput");
const summaryForm  = document.getElementById("summaryForm");
const summaryText  = document.getElementById("summaryText");
const voiceBtn     = document.getElementById("toggle-voice");
const statusEl     = document.getElementById("status");

// ---- Config ----
const SYS_PROMPT =
  "You are The Orb, a friendly AI for kids ages 6–10. Use short sentences, simple words, and a tiny emoji sometimes. Avoid scary or adult topics. If something feels unsafe, gently suggest a cheerful, safe idea instead.";
const MODEL_ID_WEBGPU = "TinyLlama/TinyLlama-1.1B-Chat-v1.0-q4f16_1";  // WebLLM model
const MAX_OUT_CHARS   = 480;
const TEMP = 0.6, TOP_P = 0.9;

// ---- State ----
let engine   = null; // WebLLM engine
let wasmPipe = null; // transformers.js pipeline
let history  = [];
let useVoice = false, speaking = false;

// ---- Helpers ----
function addBubble(text, who = "orb") {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.innerText = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
function addThinking() {
  const div = document.createElement("div");
  div.className = "bubble orb thinking";
  div.innerText = "✨ thinking…";
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}
function updateBubble(div, text) { div.innerText = text; chat.scrollTop = chat.scrollHeight; }
function setStatus(msg){ if (statusEl) statusEl.textContent = msg; }
function clampReply(s) {
  if (!s) return s;
  if (s.length > MAX_OUT_CHARS) s = s.slice(0, MAX_OUT_CHARS).replace(/\s+\S*$/,"") + "…";
  return s.split(/(?<=[.!?])\s+/).slice(0,4).join(" ");
}
function kidSafe(t){
  return !/\b(violence|weapon|self[- ]?harm|suicide|drugs?|alcohol|blood|gore|terror|extrem|sex|porn|nudity|hate|racis|homoph|xenoph|slur)\b/i.test(t||"");
}
function mysteryHintLine() {
  const hints = [
    "I followed star crumbs!",
    "I juggled clues in a maze!",
    "I danced with the numbers!",
    "I sailed across idea islands!"
  ];
  return " " + hints[Math.floor(Math.random()*hints.length)];
}

// ---- Voice (optional) ----
function speak(text){
  if (!useVoice || !window.speechSynthesis) return;
  if (speaking) window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  speaking = true; u.onend = ()=> speaking=false;
  window.speechSynthesis.speak(u);
}
voiceBtn?.addEventListener("click", ()=>{
  useVoice = !useVoice;
  voiceBtn.textContent = useVoice ? "🔇 Voice" : "🎙️ Voice";
  if (!useVoice && window.speechSynthesis) window.speechSynthesis.cancel();
});

// Optional Chrome dictation (long-press input)
if ("webkitSpeechRecognition" in window) {
  let rec;
  input.addEventListener("pointerdown", (e)=>{
    if (e.pressure < 0.5) return;
    rec = new webkitSpeechRecognition();
    rec.lang="en-US"; rec.interimResults=false; rec.maxAlternatives=1;
    rec.onresult = ev => input.value = ev.results[0][0].transcript;
    rec.start();
  });
}

// ---- INIT (WebGPU → WASM fallback) ----
async function initLLM(){
  setStatus("Checking your device…");

  // Fast path: WebGPU + WebLLM
  if ('gpu' in navigator && window.webllm?.CreateMLCEngine) {
    try {
      setStatus("Loading tiny model (WebGPU). First time can take a bit…");
      engine = await webllm.CreateMLCEngine({ model: MODEL_ID_WEBGPU, temperature: TEMP, top_p: TOP_P });
      setStatus("Ready! Ask me anything cheerful. ✨");
      return;
    } catch (e) {
      console.warn("WebLLM init failed, will try WASM fallback:", e);
    }
  }

  // Universal fallback: WASM + transformers.js
  try {
    if (!window.transformers?.pipeline) throw new Error("Transformers.js not loaded");
    setStatus("Using universal fallback (no GPU). Might be slower on first reply…");
    const { pipeline } = window.transformers;
    // Use a very small, widely-cached model; keep outputs short.
    wasmPipe = await pipeline("text-generation", "Xenova/distilgpt2");
    setStatus("Ready on fallback mode! ✨");
  } catch (e) {
    console.error("WASM fallback failed:", e);
    setStatus("Your browser cannot run the local model. Please try on a desktop browser.");
  }
}

// ---- Chat (streams on WebLLM; single-shot on WASM) ----
async function chatOnce(userText, {injectMystery=true} = {}){
  if (!kidSafe(userText)) {
    const msg = "I can’t talk about that. Let’s try space bugs, rainbow planets, or tiny robots instead! 🤖✨";
    addBubble(msg,"orb"); speak(msg); return;
  }

  addBubble(userText, "kid");
  const thinking = addThinking();

  const maybeMystery = (mystery.checked && injectMystery) ? "\n\nAdd one playful mystery hint in one short sentence." : "";
  const messages = [{ role:"system", content: SYS_PROMPT }, ...history, { role:"user", content: userText + maybeMystery }];

  try {
    let reply = "";

    if (engine) {
      // WebLLM streaming path
      let buf = "";
      const streamCb = (delta) => { buf += delta; updateBubble(thinking, buf); };
      await engine.chat.completions.create({ messages, stream:true }, streamCb);
      reply = buf.trim();
    } else if (wasmPipe) {
      // WASM single-shot path (short output)
      const prompt = `${SYS_PROMPT}\n\nUser: ${userText}\nAssistant:`;
      const out = await wasmPipe(prompt, {
        max_new_tokens: 60,
        temperature: 0.9,
        top_p: 0.95,
        do_sample: true,
        repetition_penalty: 1.1
      });
      reply = (out?.[0]?.generated_text || "").split("Assistant:").pop().trim();
      if (mystery.checked && injectMystery) reply += mysteryHintLine();
      updateBubble(thinking, reply);
    } else {
      updateBubble(thinking, "I couldn’t start my local brain here. Please try on a desktop browser.");
      return;
    }

    reply = clampReply(reply);
    updateBubble(thinking, reply);
    history.push({role:"user", content:userText}, {role:"assistant", content:reply});
    speak(reply);
  } catch (e) {
    console.error("chat error:", e);
    updateBubble(thinking, "We’re stuck in cosmic traffic. Please try again!");
  }
}

// ---- Wire up forms ----
form.addEventListener("submit", (e)=>{
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  chatOnce(q, {injectMystery:true});
});

fairnessForm.addEventListener("submit", (e)=>{
  e.preventDefault();
  const scenario = fairnessInput.value.trim();
  if (!scenario) return;
  fairnessInput.value = "";
  const prompt =
    "You are The Orb. The child is exploring what fairness means. " +
    "Answer briefly (3 short sentences). Then ask one question that helps the child think about kindness and fair rules. " +
    "Avoid judging people; talk about fair methods like taking turns, random choice, or practice.\n\nChild's scenario: " + scenario;
  chatOnce(prompt, {injectMystery:false});
});

creativeForm.addEventListener("submit", (e)=>{
  e.preventDefault();
  const idea = creativeInput.value.trim();
  if (!idea) return;
  creativeInput.value = "";
  const prompt =
    "You are The Orb. The child is imagining a fun world. " +
    "Invent whimsical details that are safe and silly. Keep it short (1–4 short sentences). Include one tiny ASCII doodle.\n\n" +
    "Child's idea: " + idea;
  chatOnce(prompt, {injectMystery:false});
});

summaryForm.addEventListener("submit", (e)=>{
  e.preventDefault();
  const txt = summaryText.value.trim();
  if (!txt) return;
  const mode = document.querySelector("input[name='mode']:checked")?.value || "one_paragraph";
  const styleHint = mode === "bullets" ? "Make 3 short bullet points using dashes."
                  : mode === "eli5"   ? "Explain like I'm 5 years old using simple words and a playful example."
                                      : "One short paragraph (3–4 short sentences).";
  const prompt =
    "You are The Orb. Make a kid-friendly summary of this text. " + styleHint +
    "\n\nText:\n" + txt + "\n\nOutput:";
  chatOnce(prompt, {injectMystery:false});
});

// ---- Boot ----
initLLM();
