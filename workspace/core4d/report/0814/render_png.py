#!/usr/bin/env python3
"""Render spider_algorithm_comparison.html -> PNG (crisp, full-page)."""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
HTML = HERE / "spider_algorithm_comparison.html"
OUT = HERE / "spider_algorithm_comparison.png"


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--force-color-profile=srgb"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000},
                                device_scale_factor=2)
        page.goto(HTML.as_uri())
        page.wait_for_timeout(400)  # let fonts/layout settle
        # tight crop around the content (.wrap) with a small margin
        box = page.locator(".wrap").bounding_box()
        m = 22
        clip = {
            "x": max(0, box["x"] - m), "y": max(0, box["y"] - m),
            "width": box["width"] + 2 * m, "height": box["height"] + 2 * m,
        }
        page.screenshot(path=str(OUT), clip=clip)
        browser.close()
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
