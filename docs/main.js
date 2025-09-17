// Browser-only Ask the Orb using WebLLM (no backend)

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

const SYS_PROMPT =
  "You are The Orb, a friendly AI for kids ages 6–10. Use short sentences, simple words, and a tiny emoji sometimes. Avoid scary or adult topics. If something feels unsafe, gently suggest a cheerful, safe idea instead.";
const MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0-q4f16_1"; // small, browser-friendly
const TEMP = 0.6, TOP_P = 0.9, MAX_OUT_CHARS = 480;

let engine = null;
let history = [];
let useVoice = false, speaking = false;

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
function clampReply(s) {
  if (!s) return s;
  if (s.length > MAX_OUT_CHARS) s = s.slice(0, MAX_OUT_CHARS).replace(/\s+\S*$/,"") + "…";
  return s.split(/(?<=[.!?])\s+/).slice(0,4).join(" ");
}
function kidSafe(t){
  return !/\b(violence|weapon|self[- ]?harm|suicide|drugs?|alcohol|blood|gore|terror|extrem|sex|porn|nudity|hate|racis|homoph|xenoph|slur)\b/i.test(t||"");
}

// Voice (optional)
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

// Init WebLLM
async function initLLM(){
  const boot = addBubble("Loading the tiny model (first time can take a bit)…","orb");
  try {
    engine = await webllm.CreateMLCEngine({ model: MODEL_ID, temperature: TEMP, top_p: TOP_P });
    updateBubble(boot, "Ready! Ask me anything cheerful. ✨");
    console.log("WebLLM Ready:", MODEL_ID);
  } catch (e) {
    console.error("WebLLM init error:", e);
    updateBubble(boot, "Your browser might not support WebGPU. Try desktop Chrome/Edge/Safari.");
  }
}

// Chat (streaming)
async function chatOnce(userText, {injectMystery=true} = {}){
  if (!engine) { addBubble("I’m still waking up. One moment, please!","orb"); return; }
  if (!kidSafe(userText)) {
    const msg = "I can’t talk about that. Let’s try space bugs, rainbow planets, or tiny robots instead! 🤖✨";
    addBubble(msg,"orb"); speak(msg); return;
  }

  addBubble(userText, "kid");
  const thinking = addThinking();

  const mysteryHint = (mystery.checked && injectMystery)
    ? "\n\nAdd one playful mystery hint in one short sentence."
    : "";

  const messages = [
    { role:"system", content: SYS_PROMPT },
    ...history,
    { role:"user", content: userText + mysteryHint }
  ];

  let reply = "";
  const streamCb = (delta) => { reply += delta; updateBubble(thinking, reply); };

  try {
    await engine.chat.completions.create({ messages, stream:true }, streamCb);
    reply = clampReply((reply||"").trim());
    updateBubble(thinking, reply);
    history.push({role:"user", content:userText}, {role:"assistant", content:reply});
    speak(reply);
  } catch (e) {
    console.error("chat error:", e);
    updateBubble(thinking, "Oops, I got tangled in star wires. Please try again!");
  }
}

// Wire up forms (prompt-based)
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

// Boot
initLLM();
