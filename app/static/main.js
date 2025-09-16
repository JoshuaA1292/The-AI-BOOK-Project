// Simple front-end client for Ask the Orb
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

let useVoice = false;
let speaking = false;

// --- Helpers ---
function addBubble(text, who = "orb") {
  const div = document.createElement("div");
  div.className = `bubble ${who}`;
  div.innerText = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function addThinking() {
  const div = document.createElement("div");
  div.className = "bubble orb thinking";
  div.innerText = "✨ thinking…";
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let msg = "Oops, something went wrong.";
    try {
      const data = await res.json();
      if (data && data.error) msg = data.error;
    } catch {}
    throw new Error(msg);
  }
  return res.json();
}

// --- Voice (optional, graceful fallback) ---
function speak(text) {
  if (!useVoice || !window.speechSynthesis) return;
  if (speaking) window.speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  speaking = true;
  u.onend = () => (speaking = false);
  window.speechSynthesis.speak(u);
}

voiceBtn?.addEventListener("click", () => {
  useVoice = !useVoice;
  voiceBtn.textContent = useVoice ? "🔇 Voice" : "🎙️ Voice";
  if (!useVoice && window.speechSynthesis) window.speechSynthesis.cancel();
});

// Optional: microphone input (Web Speech API – Chrome only)
if ("webkitSpeechRecognition" in window) {
  // long-press on the input to dictate
  let rec;
  input.addEventListener("pointerdown", (e) => {
    if (e.pressure < 0.5) return;
    rec = new webkitSpeechRecognition();
    rec.lang = "en-US";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onresult = (ev) => {
      const txt = ev.results[0][0].transcript;
      input.value = txt;
    };
    rec.start();
  });
}

// --- Chat form ---
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q) return;

  addBubble(q, "kid");
  input.value = "";

  const thinking = addThinking();
  try {
    const data = await postJSON("/api/ask_orb", {
      question: q,
      mystery_mode: mystery.checked,
    });
    thinking.remove();
    addBubble(data.text, "orb");
    speak(data.text);
  } catch (err) {
    thinking.remove();
    addBubble(String(err.message || err), "orb");
  }
});

// --- Fairness tool ---
fairnessForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const scenario = fairnessInput.value.trim();
  if (!scenario) return;
  addBubble(`(Fairness) ${scenario}`, "kid");
  fairnessInput.value = "";
  const thinking = addThinking();
  try {
    const data = await postJSON("/api/fairness_test", { scenario });
    thinking.remove();
    addBubble(data.text, "orb");
    speak(data.text);
  } catch (err) {
    thinking.remove();
    addBubble(String(err.message || err), "orb");
  }
});

// --- Creative world tool ---
creativeForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const idea = creativeInput.value.trim();
  if (!idea) return;
  addBubble(`(Create) ${idea}`, "kid");
  creativeInput.value = "";
  const thinking = addThinking();
  try {
    const data = await postJSON("/api/creative_world", { idea });
    thinking.remove();
    addBubble(data.text, "orb");
    speak(data.text);
  } catch (err) {
    thinking.remove();
    addBubble(String(err.message || err), "orb");
  }
});

// --- Summarize tool ---
summaryForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const txt = summaryText.value.trim();
  if (!txt) return;
  const mode = document.querySelector("input[name='mode']:checked")?.value || "one_paragraph";
  addBubble("(Summarize) Okay! Sending a short passage…", "kid");
  const thinking = addThinking();
  try {
    const data = await postJSON("/api/summarize", { text: txt, mode });
    thinking.remove();
    addBubble(data.text, "orb");
    speak(data.text);
  } catch (err) {
    thinking.remove();
    addBubble(String(err.message || err), "orb");
  }
});
