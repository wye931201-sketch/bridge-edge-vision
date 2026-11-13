"""Honest, legible result cards, including explicit no-detection states."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .taxonomy import BY_ID

EXPECTED_LABELS = {"裂缝": "crack", "剥落": "spalling", "混凝土剥落": "spalling", "锈蚀": "rust",
                   "锈蚀 / 锈迹": "rust", "露筋": "exposed_rebar", "蜂窝麻面": "rockpocket", "划痕": "scratches"}
STATUS_TEXT = {"detected": "检出候选 · 请复核", "target_missed": "目标类别未检出 · 需要复核",
               "empty": "未检出任何候选 · 需要复核"}
STATUS_COLORS = {"detected": (12, 109, 100), "target_missed": (161, 92, 12), "empty": (174, 45, 58)}


def result_state(detections, expected_label=None):
    if not detections:
        status = "empty"
    elif expected_label and not any(item["label"] == expected_label for item in detections):
        status = "target_missed"
    else:
        status = "detected"
    return {"status": status, "status_text": STATUS_TEXT[status], "finding_count": len(detections),
            "expected_label": expected_label,
            "target_detected": any(item["label"] == expected_label for item in detections) if expected_label else None,
            "detected_labels": sorted({item["label"] for item in detections})}


def _font(size):
    # CJK system fonts make local Windows output readable; Linux has fallbacks.
    choices = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "DejaVuSans.ttf")
    for path in choices:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _fit(draw, text, font, width):
    if draw.textbbox((0, 0), text, font=font)[2] <= width:
        return text
    while text and draw.textbbox((0, 0), text + "…", font=font)[2] > width:
        text = text[:-1]
    return text + "…"


def render_result_card(image, detections, category=None, model="", confidence=.15, profile="standard", palette=None, crack_mask=None):
    """Add a separate status header; only actual predictions mark the photo."""
    expected = EXPECTED_LABELS.get(category)
    state = result_state(detections, expected)
    source_height, source_width = image.shape[:2]
    # Small industrial photographs need a legible display, not a changed model input.
    scale = max(1., 800 / source_width)
    width, height = round(source_width * scale), round(source_height * scale)
    photograph = Image.fromarray(image[:, :, ::-1]).resize((width, height), Image.Resampling.BICUBIC)
    overlay = Image.new("RGBA", photograph.size, (0, 0, 0, 0))
    mask_pixels = 0
    if crack_mask is not None:
        mask = np.asarray(crack_mask)
        if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
            raise ValueError("Expected a binary predicted crack mask")
        mask_pixels = int(mask.sum())
        tint = Image.new("RGBA", photograph.size, (245, 99, 77, 0))
        tint.putalpha(Image.fromarray(mask.astype(np.uint8) * 112).resize(photograph.size, Image.Resampling.NEAREST))
        overlay = Image.alpha_composite(overlay, tint)
    drawing = ImageDraw.Draw(overlay)
    label_font = _font(max(22, min(44, width // 48)))
    stroke = max(4, round(min(width, height) / 200))
    for index, detection in enumerate(detections, 1):
        box = [float(value) * scale for value in detection["xyxy"]]
        if len(box) != 4 or not 0 <= box[0] < box[2] <= width + .1 or not 0 <= box[1] < box[3] <= height + .1:
            raise ValueError("Cannot render an invalid detection box")
        color = palette[detection["class_id"]] if palette is not None else BY_ID[detection["class_id"]].color
        drawing.rectangle(box, fill=(*color, 32), outline=(*color, 255), width=stroke)
        chinese = detection.get("chinese", detection["label"])
        method = "·分割" if detection.get("proposal_source") == "upstream_crack_segmentation" else ""
        text = f"{index}. {chinese}{method} {detection['confidence']:.2f}"
        bounds = drawing.textbbox((0, 0), text, font=label_font)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x = min(max(0., box[0]), max(0, width - text_width - 16))
        y = max(0., box[1] - text_height - 16)
        drawing.rectangle((x, y, x + text_width + 16, y + text_height + 12), fill=(*color, 255))
        drawing.text((x + 8, y + 4 - bounds[1]), text, font=label_font, fill=(15, 23, 42, 255))
    photograph = Image.alpha_composite(photograph.convert("RGBA"), overlay).convert("RGB")
    header_height = max(156, min(240, width // 7))
    canvas = Image.new("RGB", (width, height + header_height), (246, 248, 250))
    canvas.paste(photograph, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    color = STATUS_COLORS[state["status"]]
    draw.rectangle((0, 0, 12, header_height - 1), fill=color)
    title = _font(max(26, min(48, width // 32)))
    body = _font(max(18, min(30, width // 50)))
    draw.text((28, 12), state["status_text"], font=title, fill=color)
    actual = "、".join(sorted({item.get("chinese", item["label"]) for item in detections})) or "无"
    lines = [f"照片分组：{category or '自选图片'}   实际候选：{len(detections)} 个   类别：{actual}",
             f"模型：{model}   阈值：{confidence:g}   方案：{profile}",
             "无框表示未检出，不代表没有病害；照片分组来自源标注，不参与模型推理。"]
    line_height = max(28, (header_height - 68) // 3)
    for index, text in enumerate(lines):
        draw.text((28, 66 + index * line_height), _fit(draw, text, body, width - 52), font=body, fill=(45, 58, 70))
    metadata = {**state, "source_size": [source_width, source_height], "rendered_size": list(canvas.size),
                "display_scale": scale, "photo_origin": [0, header_height], "header_height": header_height,
                "annotation_source": "actual model predictions only"}
    metadata["predicted_mask_native_pixels"] = mask_pixels
    return canvas, metadata
