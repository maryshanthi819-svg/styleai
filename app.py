"""
Style AI — app.py
Pure Flask app. No Streamlit. Run with: python app.py
"""

import os, re, base64, urllib.parse, json
import cv2, numpy as np
from groq import Groq
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, redirect

# ── Load env ──────────────────────────────────────────────────
# override=False means Railway/system env vars take priority over .env file
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=_env_path, override=False)

# ── App setup ─────────────────────────────────────────────────
app = Flask(__name__, static_folder='static')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_groq():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key and key != "your_groq_api_key_here":
        return Groq(api_key=key)
    return None


# ══════════════════════════════════════════════════════════════
#  HTML ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def root():
    return redirect("/login")

@app.route("/login")
@app.route("/login.html")
def login_page():
    return send_from_directory(BASE_DIR, "login.html")

@app.route("/index")
@app.route("/index.html")
def index_page():
    return send_from_directory(BASE_DIR, "index.html")


# ══════════════════════════════════════════════════════════════
#  SKIN TONE DETECTION
# ══════════════════════════════════════════════════════════════

def detect_skin_tone(img_bytes):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return "Medium", 180, 140, 110
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Try face detection first for accurate skin sample
        try:
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                # Sample forehead area — more accurate than centre face
                fx = x + w // 4
                fy = y + int(h * 0.1)
                fw = w // 2
                fh = int(h * 0.25)
                sample = img_rgb[fy:fy+fh, fx:fx+fw]
                if sample.size == 0:
                    sample = img_rgb[y:y+h, x:x+w]
            else:
                h, w = img_rgb.shape[:2]
                sample = img_rgb[h//4 : 3*h//4, w//4 : 3*w//4]
        except Exception:
            h, w = img_rgb.shape[:2]
            sample = img_rgb[h//4 : 3*h//4, w//4 : 3*w//4]

        avg = sample.mean(axis=(0, 1))
        r, g, b = int(avg[0]), int(avg[1]), int(avg[2])

        # Use HSV lightness for reliable skin tone classification
        hsv_pixel  = np.array([[[r, g, b]]], dtype=np.uint8)
        hsv        = cv2.cvtColor(hsv_pixel, cv2.COLOR_RGB2HSV)[0][0]
        brightness = int(hsv[2])  # V channel 0-255

        if brightness > 210:
            tone = "Fair"
        elif brightness > 180:
            tone = "Light"
        elif brightness > 148:
            tone = "Medium"
        elif brightness > 110:
            tone = "Tan"
        elif brightness > 75:
            tone = "Brown"
        else:
            tone = "Deep"

        return tone, r, g, b
    except Exception:
        return "Medium", 180, 140, 110


# ══════════════════════════════════════════════════════════════
#  PARSE GROQ OUTPUT
# ══════════════════════════════════════════════════════════════

def parse_recs(raw):
    sections, current, items = {}, None, []
    keys = [
        "DRESS_CODE", "SUGGESTED_OUTFIT", "SHIRT_DETAILS", "PANT_DETAILS",
        "SHOES_DETAILS", "HAIRSTYLE", "ACCESSORIES", "COLOR_PALETTE", "WHY_IT_WORKS"
    ]

    def normalise(s):
        return s.upper().replace(" ", "_").replace("-", "_").replace("/", "_")

    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        matched = next((k for k in keys if k in normalise(line)), None)
        if matched:
            if current:
                sections[current] = items
            current, items = matched, []
        elif current:
            clean = line.lstrip("→-*•·").strip()
            clean = re.sub(r"^\d+[.)]\s*", "", clean)
            if clean and normalise(clean) not in keys:
                items.append(clean)

    if current:
        sections[current] = items
    return sections


# ══════════════════════════════════════════════════════════════
#  BUILD SHOPPING LINKS
# ══════════════════════════════════════════════════════════════

def build_shopping_from_recs(rec, tone, gender, occasion):

    def best_platform(name):
        nl = name.lower()
        accessories = ["watch", "chain", "bracelet", "necklace", "ring",
                       "earring", "bag", "clutch", "belt", "sunglass", "scarf",
                       "wallet", "cap", "hat", "tie", "cufflink", "jhumka",
                       "jhumki", "bangle", "bangles"]
        if any(w in nl for w in accessories):
            return "Amazon.in"
        footwear = ["shoe", "boot", "loafer", "heel", "sandal", "sneaker",
                    "derby", "oxford", "chelsea", "mule", "stiletto", "wedge"]
        if any(w in nl for w in footwear):
            return "Amazon.in"
        return "Flipkart"

    def make_url(platform, query):
        q = urllib.parse.quote_plus(query)
        if platform == "Amazon.in":
            return f"https://www.amazon.in/s?k={q}"
        return f"https://www.flipkart.com/search?q={q}"

    icon_map = {
        "shirt": "👕", "top": "👕", "polo": "👕", "tshirt": "👕", "t-shirt": "👕",
        "kurti": "👗", "kurta": "👘", "saree": "👘", "anarkali": "👗", "lehenga": "👗",
        "blouse": "👗", "dress": "👗", "suit": "👔", "blazer": "🧥", "jacket": "🧥",
        "pant": "👖", "trouser": "👖", "jeans": "👖", "legging": "👖",
        "palazzo": "👖", "chino": "👖",
        "shoe": "👟", "boot": "🥾", "loafer": "🥿", "heel": "👠", "stiletto": "👠",
        "sandal": "👡", "sneaker": "👟", "derby": "👞", "oxford": "👞",
        "chelsea": "🥾", "wedge": "👡",
        "watch": "⌚", "chain": "🏆", "bracelet": "💛", "necklace": "💍",
        "earring": "💎", "jhumka": "💎", "jhumki": "💎", "ring": "💍",
        "bangle": "🔮", "bangles": "🔮",
        "bag": "👜", "clutch": "👛", "belt": "🔗", "sunglass": "🕶️",
        "scarf": "🧣", "cap": "🧢",
    }

    def get_icon(name):
        nl = name.lower()
        for k, v in icon_map.items():
            if k in nl:
                return v
        return "🛍️"

    def extract_name(lines, fallback):
        color = type_ = fabric = style = fit = ""
        for line in lines:
            lw  = line.lower()
            val = line.split(":", 1)[1].strip() if ":" in line else ""
            if lw.startswith(("color:", "colour:")):  color  = val
            elif lw.startswith("type:"):              type_  = val
            elif lw.startswith("fabric:"):            fabric = val
            elif lw.startswith("style:"):             style  = val
            elif lw.startswith("fit:"):               fit    = val
        parts = []
        if color:  parts.append(color)
        if fit:    parts.append(fit)
        elif style: parts.append(style)
        if type_:  parts.append(type_)
        if fabric: parts.append(fabric)
        if parts:
            return " ".join(parts)
        if lines:
            first = lines[0]
            return first.split(":", 1)[1].strip() if ":" in first else first
        return fallback

    products = []

    for section, fallback in [("SHIRT_DETAILS", "top"), ("PANT_DETAILS", "bottom"),
                               ("SHOES_DETAILS", "footwear")]:
        lines = rec.get(section, [])
        if lines:
            name = extract_name(lines, fallback)
            p    = best_platform(name)
            products.append({"name": name, "platform": p,
                             "url": make_url(p, name), "icon": get_icon(name)})

    for raw_acc in rec.get("ACCESSORIES", [])[:3]:
        clean = re.sub(r"^\d+[.)]\s*", "", raw_acc)
        name  = re.sub(r"\s*\(.*?\)\s*$", "", clean.split("—")[0].split("–")[0]).strip()
        if name:
            p = best_platform(name)
            products.append({"name": name, "platform": p,
                             "url": make_url(p, name), "icon": get_icon(name)})

    return products


# ══════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data      = request.json or {}
        image_b64 = data.get("image", "")
        gender    = data.get("gender", "Male")
        occasion  = data.get("occasion", "Casual")
        prompt    = data.get("prompt", "")

        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)

        tone, r, g, b = detect_skin_tone(img_bytes)

        client = get_groq()
        if not client:
            return jsonify({"success": False,
                            "error": "GROQ_API_KEY missing. Add it to your .env file."})

        style_ctx = f"Occasion: {occasion}. Extra style notes: {prompt}" if prompt \
                    else f"Occasion: {occasion}."

        import random
        palette_seed = random.choice([
            "earthy tones", "jewel tones", "pastel tones", "monochromatic", "bold contrasts",
            "muted neutrals", "warm tones", "cool tones", "festive Indian colors", "minimal palette"
        ])

        ai_prompt = f"""You are an expert fashion stylist. A {gender} client has a {tone} skin tone (RGB: {r},{g},{b}).
{style_ctx}

IMPORTANT RULES:
- Output ALL sections. Never skip any.
- Use EXACTLY this format with → prefix.
- Color palette MUST use {palette_seed} — pick 3 UNIQUE specific colors that are DIFFERENT from each other.
- Every suggestion must be SPECIFICALLY tailored to {tone} skin tone with RGB ({r},{g},{b}).
- Never repeat the same palette across sessions — be creative and vary your choices.

DRESS_CODE
→ [dress code type]

SUGGESTED_OUTFIT
→ [Complete head-to-toe outfit in one vivid sentence tailored to {tone} skin tone]

COLOR_PALETTE
→ Primary: [specific color that flatters {tone} skin — NOT generic, e.g. "Burnt Sienna", "Cobalt Blue", "Forest Green"]
→ Secondary: [different specific color, complementary to primary]
→ Accent: [third specific color, contrasting pop]

SHIRT_DETAILS
→ Color: [specific color]
→ Type: [exact shirt type e.g. "Linen Button-Down Shirt", "Silk Kurti"]
→ Fit: [fit e.g. "Slim Fit", "Relaxed Fit"]
→ Fabric: [fabric type]

PANT_DETAILS
→ Color: [specific color]
→ Type: [exact pant type e.g. "Slim-Fit Chinos", "High-Waist Palazzo"]
→ Fit: [fit]
→ Fabric: [fabric]

SHOES_DETAILS
→ Color: [color]
→ Type: [exact shoe type e.g. "White Leather Derby Shoes", "Tan Suede Loafers"]

ACCESSORIES
→ 1. [accessory name] — [why it works]
→ 2. [accessory name] — [why it works]
→ 3. [accessory name] — [why it works]

HAIRSTYLE
→ Style: [specific hairstyle name]
→ How-to: [one maintenance tip]
→ Products: [recommended hair product]
→ Tip: [pro styling tip]

WHY_IT_WORKS
→ [2 sentences: why these choices specifically complement {tone} skin tone with RGB {r},{g},{b}]

Be specific, creative and consider Indian fashion. Output ONLY the format above — no extra text."""

        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": ai_prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=2000,
            temperature=0.9,
        )
        raw      = resp.choices[0].message.content
        print("=== GROQ RAW OUTPUT ===")
        print(raw)
        print("=== PARSED ===")
        parsed   = parse_recs(raw)
        print(parsed)
        print("=== TONE:", tone, "RGB:", r, g, b)
        products = build_shopping_from_recs(parsed, tone, gender, occasion)

        return jsonify({
            "success": True,
            "skin_tone": tone,
            "rgb": {"r": r, "g": g, "b": b},
            "recommendations": parsed,
            "products": products,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/suggestions", methods=["GET"])
def suggestions():
    try:
        client = get_groq()
        if not client:
            return jsonify({"suggestions": _fallback_suggestions()})

        resp = client.chat.completions.create(
            messages=[{"role": "user", "content": (
                "Generate exactly 6 short, interesting fashion questions an Indian user might ask "
                "a style AI assistant. Cover different topics: skin tone, occasion dressing, "
                "Indian fashion, accessories, hairstyle, colour theory. "
                "Output ONLY a JSON array of 6 strings, nothing else. No extra text."
            )}],
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            temperature=0.9,
        )
        raw  = resp.choices[0].message.content.strip()
        raw  = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        chips = json.loads(raw)
        if isinstance(chips, list) and len(chips) >= 4:
            return jsonify({"suggestions": chips[:6]})
        return jsonify({"suggestions": _fallback_suggestions()})
    except Exception:
        return jsonify({"suggestions": _fallback_suggestions()})


def _fallback_suggestions():
    return [
        "What colours suit an olive skin tone?",
        "What to wear to an Indian wedding as a guest?",
        "Best accessories for casual streetwear?",
        "How to style a saree for office?",
        "What hairstyle suits a round face?",
        "What are the latest Indian fashion trends?",
    ]


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data    = request.json or {}
        message = data.get("message", "")
        history = data.get("history", [])

        client = get_groq()
        if not client:
            return jsonify({"reply": "⚠️ GROQ_API_KEY missing. Add it to your .env file."})

        msgs = [{"role": "system", "content": (
            "You are a witty, knowledgeable Indian fashion assistant for Style AI. "
            "You specialise in outfit pairing, colour theory, occasion dressing, "
            "and Indian/global fashion trends. Keep replies concise (under 4 sentences) "
            "unless the user asks for a detailed breakdown."
        )}]
        for m in history[-6:]:
            msgs.append(m)
        msgs.append({"role": "user", "content": message})

        resp = client.chat.completions.create(
            messages=msgs,
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            temperature=0.8,
        )
        return jsonify({"reply": resp.choices[0].message.content})

    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})


# ══════════════════════════════════════════════════════════════
#  RUN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n  Style AI running → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
