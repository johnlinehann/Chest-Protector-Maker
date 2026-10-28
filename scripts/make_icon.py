"""Regenerate assets/chestforge.{png,ico,icns}. Needs Pillow (dev only; the installer does not)."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).resolve().parent.parent / "assets"
S = 1024


def draw() -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((40, 40, S - 40, S - 40), radius=200, fill=(22, 27, 38, 255))

    # Chest-protector plate: shoulders, collar notch, tapered body, segmented abdomen
    plate = [(250, 230), (420, 210), (460, 290), (564, 290), (604, 210), (774, 230),
             (800, 470), (760, 760), (512, 850), (264, 760), (224, 470)]
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon(plate, fill=(255, 140, 40, 150))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(28)))
    d.polygon(plate, fill=(58, 66, 84, 255), outline=(255, 150, 50, 255), width=18)
    for y in (640, 730):
        d.line((280, y, 744, y), fill=(255, 150, 50, 255), width=12)

    # Cardiac silhouette: heart, offset to the wearer's left as on the NTS (ND200 Fig. 1)
    cx, cy, r = 560, 450, 62
    d.ellipse((cx - 2 * r, cy - r, cx, cy + r), fill=(230, 45, 60, 255))
    d.ellipse((cx, cy - r, cx + 2 * r, cy + r), fill=(230, 45, 60, 255))
    d.polygon([(cx - 2 * r + 6, cy + 18), (cx + 2 * r - 6, cy + 18), (cx, cy + 2.6 * r)], fill=(230, 45, 60, 255))

    # Incoming ball + impact rings
    d.ellipse((150, 130, 290, 270), fill=(245, 245, 240, 255))
    d.arc((170, 150, 270, 250), 300, 60, fill=(220, 40, 40, 255), width=8)
    for rr in (170, 215):
        d.arc((cx - rr, cy - rr + 40, cx + rr, cy + rr + 40), 205, 245, fill=(255, 200, 120, 255), width=10)
    return img


def main():
    ASSETS.mkdir(exist_ok=True)
    img = draw()
    img.resize((512, 512), Image.LANCZOS).save(ASSETS / "chestforge.png")
    img.save(ASSETS / "chestforge.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.save(ASSETS / "chestforge.icns")


if __name__ == "__main__":
    main()
