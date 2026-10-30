"""Capture README screenshots from the running stack: uv run --with playwright python scripts/screenshots.py"""
from playwright.sync_api import sync_playwright

OUT = "docs/screenshots"
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1360, "height": 900})
    pg.goto("http://localhost:8080")
    pg.locator(".hrow").first.wait_for(timeout=30000)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=f"{OUT}/web-senior-finance.png", full_page=True)
    sels = pg.locator("select")
    sels.nth(0).select_option("CEO"); sels.nth(1).select_option("OIL"); sels.nth(2).select_option("DOCTORAL")
    pg.locator("input[type=range]").first.fill("20")
    pg.get_by_role("button", name="Industry").click()
    pg.wait_for_timeout(1500)
    pg.screenshot(path=f"{OUT}/web-ceo-oil.png", full_page=True)
    pg.goto("http://localhost:8000/docs")
    pg.get_by_text("/predict").first.wait_for()
    pg.screenshot(path=f"{OUT}/api-docs.png", full_page=True)
    b.close()
