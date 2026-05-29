from flask import Flask, request, jsonify, render_template_string
import google.generativeai as genai
from huggingface_hub import InferenceClient
from PIL import Image
import io, base64, os, uuid
from pathlib import Path
from datetime import datetime

# ==================================================
# CONFIG
# ==================================================
GEMINI_API_KEY  = "AIzaSyDEbusid02TWM1mTTSUFOIftnODGb0L7_w"
HF_TOKEN        = "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"
OUTPUT_FOLDER   = r"D:\python_aivideos\auto-hugging-face\final"

# HuggingFace — FLUX.1-dev (high quality photorealistic model)
HF_MODEL        = "black-forest-labs/FLUX.1-dev"
HF_STEPS        = 28       # 20–50 recommended for FLUX.1-dev
HF_GUIDANCE     = 3.5      # best quality range for FLUX.1-dev
IMAGE_WIDTH     = 1024
IMAGE_HEIGHT    = 1024

# ==================================================
# INIT
# ==================================================
app = Flask(__name__)

genai.configure(api_key=GEMINI_API_KEY)
prompt_model = genai.GenerativeModel("gemini-2.5-flash")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
print(f"✅ Output folder ready : {OUTPUT_FOLDER}")
print(f"✅ HF Model            : {HF_MODEL}")

# ==================================================
# HTML UI
# ==================================================
HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>AI Product Avatar Generator</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #0d0d0d;
      color: #f0f0f0;
      font-family: 'Segoe UI', Arial, sans-serif;
      padding: 40px 20px;
    }

    .container {
      max-width: 860px;
      margin: auto;
    }

    h1 {
      font-size: 1.8rem;
      margin-bottom: 6px;
      background: linear-gradient(90deg, #00c896, #00a0ff);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      color: #888;
      margin-bottom: 30px;
      font-size: 0.9rem;
    }

    .card {
      background: #1a1a1a;
      border: 1px solid #2a2a2a;
      border-radius: 14px;
      padding: 24px;
      margin-bottom: 20px;
    }

    label {
      display: block;
      margin-bottom: 8px;
      font-size: 0.85rem;
      color: #aaa;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    input[type="file"] {
      width: 100%;
      padding: 12px;
      background: #111;
      border: 1px dashed #444;
      border-radius: 8px;
      color: #ccc;
      cursor: pointer;
    }

    textarea {
      width: 100%;
      height: 180px;
      background: #111;
      color: #f0f0f0;
      border: 1px solid #2a2a2a;
      border-radius: 8px;
      padding: 14px;
      font-size: 0.9rem;
      resize: vertical;
    }

    .steps-row {
      display: flex;
      gap: 10px;
      margin-top: 14px;
    }

    .step-btn {
      flex: 1;
      padding: 10px;
      border: 1px solid #444;
      background: #111;
      color: #aaa;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
      transition: all 0.2s;
    }

    .step-btn.active {
      border-color: #00c896;
      color: #00c896;
      background: #001a13;
    }

    .btn-main {
      width: 100%;
      padding: 15px;
      margin-top: 20px;
      background: linear-gradient(90deg, #00c896, #00a0ff);
      border: none;
      color: white;
      border-radius: 10px;
      cursor: pointer;
      font-size: 1rem;
      font-weight: 600;
      letter-spacing: 0.5px;
      transition: opacity 0.2s;
    }

    .btn-main:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .status-box {
      display: none;
      background: #111;
      border: 1px solid #2a2a2a;
      border-radius: 10px;
      padding: 16px 20px;
      margin-top: 16px;
    }

    .status-step {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 6px 0;
      color: #666;
      font-size: 0.9rem;
      transition: color 0.3s;
    }

    .status-step.active  { color: #00c896; }
    .status-step.done    { color: #00ff99; }
    .status-step.error   { color: #ff5555; }

    .dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: currentColor;
      flex-shrink: 0;
    }

    .result-card {
      display: none;
      margin-top: 20px;
    }

    .result-card img {
      width: 100%;
      border-radius: 12px;
      border: 1px solid #2a2a2a;
    }

    .save-path {
      margin-top: 10px;
      font-size: 0.8rem;
      color: #666;
      word-break: break-all;
    }

    .save-path span { color: #00c896; }
  </style>
</head>
<body>
<div class="container">

  <h1>🎨 AI Product Avatar Generator</h1>
  <p class="subtitle">Upload a product image → Gemini crafts a cinematic prompt → FLUX.1-dev generates your avatar</p>

  <!-- Upload -->
  <div class="card">
    <label>1 · Upload Product Image</label>
    <input type="file" id="imageInput" accept="image/*">
  </div>

  <!-- Prompt preview -->
  <div class="card">
    <label>2 · Generated Prompt (editable)</label>
    <textarea id="promptBox" placeholder="Gemini will generate a cinematic prompt here — you can edit it before image generation…"></textarea>
  </div>

  <!-- Size selector -->
  <div class="card">
    <label>3 · Image Size</label>
    <div class="steps-row">
      <button class="step-btn active" id="btn1024" onclick="setSize(1024,1024)">⬛ 1024×1024 — Square</button>
      <button class="step-btn" id="btn1344" onclick="setSize(1344,768)">🖼 1344×768 — Landscape</button>
      <button class="step-btn" id="btn768"  onclick="setSize(768,1344)">📱 768×1344 — Portrait</button>
    </div>
  </div>

  <!-- Steps selector -->
  <div class="card">
    <label>4 · Quality (Inference Steps)</label>
    <div class="steps-row">
      <button class="step-btn" id="sBtn20" onclick="setSteps(20)">⚡ 20 Steps — Fast</button>
      <button class="step-btn active" id="sBtn28" onclick="setSteps(28)">✨ 28 Steps — Balanced</button>
      <button class="step-btn" id="sBtn50" onclick="setSteps(50)">💎 50 Steps — Max Quality</button>
    </div>
  </div>

  <button class="btn-main" id="runBtn" onclick="runPipeline()">🚀 Generate Avatar</button>

  <!-- Status -->
  <div class="status-box" id="statusBox">
    <div class="status-step" id="s1"><div class="dot"></div> Analyzing product with Gemini…</div>
    <div class="status-step" id="s2"><div class="dot"></div> Generating image with FLUX.1-dev (this takes ~30–60s)…</div>
    <div class="status-step" id="s3"><div class="dot"></div> Saving to output folder…</div>
  </div>

  <!-- Result -->
  <div class="result-card card" id="resultCard">
    <label>✅ Generated Avatar</label>
    <img id="resultImg" src="" alt="Generated avatar">
    <p class="save-path">Saved to: <span id="savePath"></span></p>
  </div>

</div>

<script>
  let imgWidth = 1024, imgHeight = 1024, imgSteps = 28;

  function setSize(w, h) {
    imgWidth = w; imgHeight = h;
    document.getElementById("btn1024").classList.toggle("active", w===1024 && h===1024);
    document.getElementById("btn1344").classList.toggle("active", w===1344);
    document.getElementById("btn768").classList.toggle("active",  w===768 && h===1344);
  }

  function setSteps(s) {
    imgSteps = s;
    document.getElementById("sBtn20").classList.toggle("active", s===20);
    document.getElementById("sBtn28").classList.toggle("active", s===28);
    document.getElementById("sBtn50").classList.toggle("active", s===50);
  }

  function setStep(id, state) {
    document.getElementById(id).className = "status-step " + state;
  }

  async function runPipeline() {
    const fileInput = document.getElementById("imageInput");
    if (fileInput.files.length === 0) { alert("Please upload a product image first."); return; }

    const runBtn = document.getElementById("runBtn");
    runBtn.disabled = true;
    document.getElementById("statusBox").style.display = "block";
    document.getElementById("resultCard").style.display = "none";
    ["s1","s2","s3"].forEach(id => setStep(id, ""));

    try {
      // ── STEP 1: Gemini prompt ────────────────────────────────
      setStep("s1", "active");
      const fd = new FormData();
      fd.append("image", fileInput.files[0]);

      const r1 = await fetch("/generate-prompt", { method: "POST", body: fd });
      const d1 = await r1.json();
      if (!d1.success) throw new Error(d1.error);
      setStep("s1", "done");

      const promptBox = document.getElementById("promptBox");
      promptBox.value = d1.prompt;

      // ── STEP 2: FLUX.1-dev image ─────────────────────────────
      setStep("s2", "active");
      const r2 = await fetch("/generate-image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptBox.value, width: imgWidth, height: imgHeight, steps: imgSteps })
      });
      const d2 = await r2.json();
      if (!d2.success) throw new Error(d2.error);

      setStep("s2", "done");
      setStep("s3", "done");

      document.getElementById("resultImg").src = "data:image/png;base64," + d2.image_b64;
      document.getElementById("savePath").textContent = d2.saved_path;
      document.getElementById("resultCard").style.display = "block";

    } catch(err) {
      console.error(err);
      alert("Error: " + err.message);
      ["s1","s2","s3"].forEach(id => {
        const el = document.getElementById(id);
        if (el.classList.contains("active")) setStep(id, "error");
      });
    } finally {
      runBtn.disabled = false;
    }
  }
</script>
</body>
</html>
"""

# ==================================================
# HOME
# ==================================================
@app.route("/")
def home():
    return render_template_string(HTML)

# ==================================================
# ROUTE 1 — Generate Prompt via Gemini
# ==================================================
@app.route("/generate-prompt", methods=["POST"])
def generate_prompt():
    try:
        print("\n=== STEP 1: GEMINI PROMPT GENERATION ===")

        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image uploaded."})

        product_image = Image.open(request.files["image"])
        print("Product image loaded.")

        analysis_prompt = """
Analyze this product image carefully.
Create a cinematic AI advertising prompt for a human avatar that will model or showcase this product.

The avatar must naturally match:
- Product category and intended use
- Product style, color palette, and aesthetic
- Target audience (age, gender, lifestyle)
- Fashion and styling aesthetic
- Mood and emotion conveyed by the product

Requirements for the avatar & scene:
- Ultra realistic human model
- Luxury commercial photography style
- Cinematic wide-angle or portrait composition
- Fashion/lifestyle magazine quality
- DSLR bokeh depth of field
- Soft cinematic directional lighting
- Realistic skin texture and detail
- Premium aspirational environment
- Photorealistic rendering
- 8K ultra detailed
- Natural, confident pose interacting with or near the product
- High-end cinematic color grading

Return ONLY the final image generation prompt. No explanations, no labels.
"""

        response = prompt_model.generate_content([analysis_prompt, product_image])
        avatar_prompt = response.text.strip()
        print(f"Prompt generated:\n{avatar_prompt}\n")

        return jsonify({"success": True, "prompt": avatar_prompt})

    except Exception as e:
        print(f"[ERROR] Gemini: {e}")
        return jsonify({"success": False, "error": str(e)})


# ==================================================
# ROUTE 2 — Generate Image via HuggingFace FLUX.1-dev
# ==================================================
@app.route("/generate-image", methods=["POST"])
def generate_image():
    try:
        print("\n=== STEP 2: FLUX.1-dev IMAGE GENERATION ===")

        data   = request.get_json()
        prompt = data.get("prompt", "").strip()
        width  = int(data.get("width",  IMAGE_WIDTH))
        height = int(data.get("height", IMAGE_HEIGHT))
        steps  = int(data.get("steps",  HF_STEPS))

        if not prompt:
            return jsonify({"success": False, "error": "Prompt is empty."})

        print(f"Model  : {HF_MODEL}")
        print(f"Size   : {width}x{height}")
        print(f"Steps  : {steps}")
        print(f"Prompt : {prompt[:100]}...")

        client = InferenceClient(model=HF_MODEL, token=HF_TOKEN)

        generated: Image.Image = client.text_to_image(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=HF_GUIDANCE,
            width=width,
            height=height,
        )

        # ── Save to disk ────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        uid       = uuid.uuid4().hex[:6]
        filename  = f"avatar_{timestamp}_{uid}.png"
        save_path = Path(OUTPUT_FOLDER) / filename
        generated.save(save_path)
        print(f"✅ Image saved → {save_path}")

        # ── Return as base64 for browser preview ────────────────
        buf = io.BytesIO()
        generated.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({
            "success":    True,
            "saved_path": str(save_path),
            "image_b64":  b64
        })

    except Exception as e:
        print(f"[ERROR] FLUX.1-dev: {e}")
        return jsonify({"success": False, "error": str(e)})


# ==================================================
# RUN
# ==================================================
if __name__ == "__main__":
    print(f"\n✅ Output folder : {OUTPUT_FOLDER}")
    print(f"✅ HF Model      : {HF_MODEL}")
    print(f"✅ Steps default : {HF_STEPS}  |  Guidance: {HF_GUIDANCE}")
    app.run(host="0.0.0.0",debug=True)
