// Ask the Orb — Browser-only main.js using WebLLM (no backend)

// --- DOM refs (kept same IDs/classes as your HTML/CSS) ---
const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const input = document.getElementById("chatInput");
const mystery = document.getElementById("mysteryMode");

const fairnessForm = document.getElementById("fairnessForm");
const fairnessInput = document.getElementById("fairnessInput");

const creativeForm = document.getElementById("creativeForm");
const creativeInput = document.getElementById("creativeInput");

const summaryForm = document.getElementById("summaryForm");
const summaryText = document.getElementById("summaryText");

const voiceBtn = document.getElementById("toggle-voice");

// --- Config ---
const SYS_PROMPT =
  "You are The Orb, a friendly AI for kids ages 6–10. Use short sentences, simple words, and a tiny emoji sometimes. Avoid scary or adult topics. If something feels unsafe, gently suggest a cheerful, safe idea instead.";
const MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0-q4f16_1"; // small, browser-friendly
const TEMP = 0.6, TOP_P = 0.9, MAX_OUT_CHARS = 480;

// --- State ---
let engine = null;
let history = []; // [{role, content}]
let useVoice = false;
let speaking = false;

// --- Helpers (UI) ---
function escapeHtml(s){return (s||"").replace(/[&<>"']/g, m=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" }[m]));}

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

function updateBubble(div, text) {
  div.innerText = text;
  chat.scrollTop = chat.scrollHeight;
}

function clampReply(s) {
  if (!s) return s;
  if (s.length > MAX_OUT_CHARS) s = s.slice(0, MAX_OUT_CHARS).replace(/\s+\S*$/,"") + "…";
  const parts = s.split(/(?<=[.!?])\s+/).slice(0,4);
  return parts.join(" ");
}

function kidSafe(text){
  const bad = /\b(violence|weapon|self[- ]?harm|suicide|drugs?|alcohol|blood|gore|terror|extrem|sex|porn|nudity|hate|racis|homoph|xenoph|slur)\b/i;
  return !bad.test(text || "");
}

// --- Voice ---
function speak(text) {
  if (!useVoice || !window.speechSynthesis) return;
  if (speaking) window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  speaking = true; u.onend = () => (speaking = false);
  window.speechSynthesis.speak(u);
}

voiceBtn?.addEventListener("click", () => {
  useVoice = !useVoice;
  voiceBtn.textContent = useVoice ? "🔇 Voice" : "🎙️ Voice";
  if (!useVoice && window.speechSynthesis) window.speechSynthesis.cancel();
});

// Optional Chrome dictation (unchanged)
if ("webkitSpeechRecognition" in window) {
  let rec;
  input.addEventListener("pointerdown", (e) => {
    if (e.pressure < 0.5) return;
    rec = new webkitSpeechRecognition();
    rec.lang = "en-US"; rec.interimResults = false; rec.maxAlternatives = 1;
    rec.onresult = (ev) => { input.value = ev.results[0][0].transcript; };
    rec.start();
  });
}

// --- WebLLM init ---
async function initLLM() {
  const status = addBubble("Loading the tiny model (first time can take a bit)…", "orb");
  try {
    engine = await webllm.CreateMLCEngine({ model: MODEL_ID, temperature: TEMP, top_p: TOP_P });
    updateBubble(status, "Ready! Ask me anything cheerful. ✨");
  } catch (e) {
    console.error(e);
    updateBubble(status, "Your browser might not support WebGPU. Try desktop Chrome/Edge/Safari.");
  }
}

// --- Core chat (streaming, no server) ---
async function chatOnce(userText, {injectMystery=true} = {}) {
  if (!engine) {
    const wait = addBubble("I’m still waking up. One moment, please!", "orb");
    return;
  }
  if (!kidSafe(userText)) {
    const safeMsg = "I can’t talk about that. Let’s try space bugs, rainbow planets, or tiny robots instead! 🤖✨";
    addBubble(safeMsg, "orb"); speak(safeMsg); return;
  }

  addBubble(userText, "kid");
  const thinking = addThinking();

  const maybeMystery = (mystery.checked && injectMystery)
    ? "\n\nAdd one playful mystery hint in one short sentence."
    : "";

  const messages = [
    { role: "system", content: SYS_PROMPT },
    ...history,
    { role: "user", content: userText + maybeMystery }
  ];

  let reply = "";
  const streamCb = (delta) => { reply += delta; updateBubble(thinking, reply); };

  try {
    await engine.chat.completions.create({ messages, stream: true }, streamCb);
    reply = clampReply((reply || "").trim());
    updateBubble(thinking, reply);
    history.push({role:"user", content:userText}, {role:"assistant", content:reply});
    speak(reply);
  } catch (e) {
    console.error(e);
    updateBubble(thinking, "Oops, I got tangled in star wires. Please try again!");
  }
}

// --- Wire up forms (now prompt-based, no /api calls) ---
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;
  input.value = "";
  chatOnce(q, {injectMystery:true});
});

fairnessForm.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const scenario = fairnessInput.value.trim();
  if (!scenario) return;
  fairnessInput.value = "";
  const prompt =
    "You are The Orb. The child is exploring what fairness means. " +
    "Answer briefly (3 short sentences). Then ask one question that helps the child think about kindness and fair rules. " +
    "Avoid judging people; talk about fair methods like taking turns, random choice, or practice.\n\n" +
    "Child's scenario: " + scenario;
  chatOnce(prompt, {injectMystery:false});
});

creativeForm.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const idea = creativeInput.value.trim();
  if (!idea) return;
  creativeInput.value = "";
  const prompt =
    "You are The Orb. The child is imagining a fun world. " +
    "Invent whimsical details that are safe and silly. Keep it short (1–4 short sentences). " +
    "Include one tiny ASCII doodle.\n\nChild's idea: " + idea;
  chatOnce(prompt, {injectMystery:false});
});

summaryForm.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const txt = summaryText.value.trim();
  if (!txt) return;
  const mode = document.querySelector("input[name='mode']:checked")?.value || "one_paragraph";
  const styleHint = mode === "bullets"
    ? "Make 3 short bullet points using dashes."
    : mode === "eli5"
      ? "Explain like I'm 5 years old using simple words and a playful example."
      : "One short paragraph (3–4 short sentences).";
  const prompt =
    "You are The Orb. Make a kid-friendly summary of this text. " + styleHint +
    "\n\nText:\n" + txt + "\n\nOutput:";
  chatOnce(prompt, {injectMystery:false});
});

// --- Boot ---
initLLM();
