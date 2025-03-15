from PIL import Image, ImageDraw, ImageFont

EMOJI_MAP = {
    "apple": "🍎",
    "car": "🚗",
    "house": "🏠",
    "tree": "🌳",
    "dog": "🐶",
    "cat": "🐱",
    "bicycle": "🚲",
    "flower": "🌸",
    "boat": "⛵",
    "star": "⭐"
}
# 尝试完整路径
font_path = "C:/Windows/Fonts/seguiemj.ttf"  # 或 seguiemoji.ttf
font_size = 72
emoji_char = EMOJI_MAP["bicycle"]

image = Image.new("RGB", (200, 200), (255, 255, 255))
draw = ImageDraw.Draw(image)

try:
    emoji_font = ImageFont.truetype(font_path, font_size)
except Exception as e:
    print("Load emoji font failed:", e)
    emoji_font = ImageFont.load_default()

# 在(50, 50)绘制
draw.text((50, 50), emoji_char, font=emoji_font, fill="black")

image.show()
image.save("test_emoji.png")
