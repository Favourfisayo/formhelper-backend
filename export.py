from dotenv import load_dotenv
import os, io, json
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont


# Step 0: Init Gemini client
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)


def fit_text_to_box(draw, text, box, font_path="arial.ttf", max_size=28, min_size=10):
    """Scale text to fit within the given box dimensions."""
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1

    for size in range(max_size, min_size, -1):
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        if text_w <= box_w * 0.95 and text_h <= box_h * 0.9:
            return font
    return ImageFont.truetype(font_path, min_size)


def export_form(image_bytes: bytes, form_fields: list, output_pdf: str = "filled_form.pdf") -> str:
    """
    Fills a form image with provided field values and returns path to generated PDF.
    - image_bytes: bytes of the form image (downloaded from Supabase)
    - form_fields: [{"label": "Name", "value": "John"}]
    - output_pdf: filename to save
    """

# Step 1: Load image
    base_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = base_image.size


# Step 2: Detect fields via Gemini
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    prompt = """
    Detect and parse very well only input fields (text boxes, signature areas, lines, empty spaces) in this form.
    Return a JSON list where each entry has:
      - "label": the field label or description
      - "box_2d": [ymin, xmin, ymax, xmax] normalized 0-1000
    Include all fields even if labels are missing.
    """
    config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0), #thinking_budget to 0 for better results in object detection, according to gemini's docs
    response_mime_type="application/json"
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image_part, prompt],
        config=config
    )

    fields_detected = json.loads(response.text)


# Step 3: Normalize coords
    for f in fields_detected:
        if f is None or "label" not in f or f["label"] is None:
            print("Invalid field found:", f)
        else:
            print("Field label:", f["label"])
    field_map = {}
    for f in fields_detected:
        ymin, xmin, ymax, xmax = f["box_2d"]
        abs_x1 = int(xmin / 1000 * width)
        abs_y1 = int(ymin / 1000 * height)
        abs_x2 = int(xmax / 1000 * width)
        abs_y2 = int(ymax / 1000 * height)
        field_map[f["label"].lower()] = [abs_x1, abs_y1, abs_x2, abs_y2]


# Step 4: Draw overlay
    overlay = Image.new("RGBA", base_image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for field in form_fields:
        label = field["label"].lower()
        value = str(field["value"]).strip()

        if label in field_map and value:
            x1, y1, x2, y2 = field_map[label]
            font = fit_text_to_box(draw, value, (x1, y1, x2, y2))
            bbox = draw.textbbox((0, 0), value, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            box_w, box_h = x2 - x1, y2 - y1

            text_x = x1 + (box_w - text_w) // 2
            text_y = y1 + (box_h - text_h) // 2 - 2
            draw.text((text_x, text_y), value, fill="black", font=font)
        else:
            print(f"⚠️ Warning: Field '{label}' not detected or empty.")

    # Step 5: Merge & Save
    filled_form = Image.alpha_composite(base_image.convert("RGBA"), overlay)
    filled_form.convert("RGB").save(output_pdf, "PDF")

    return output_pdf
