from playwright.sync_api import Page, expect, sync_playwright
import time

def verify_gui(page: Page):
    # 1. Go to the app
    page.goto("http://localhost:3000")

    # 2. Assert title or content to ensure it loaded
    expect(page.get_by_role("heading", name="Avantime Ray Tracer")).to_be_visible()

    # 3. Find the render button and click it
    render_button = page.get_by_role("button", name="Render Scene")
    render_button.click()

    # 4. Wait for the image to appear.
    # The image is only shown when imageUrl is present.
    # It has alt="Rendered Scene"
    img = page.get_by_alt_text("Rendered Scene")
    expect(img).to_be_visible(timeout=30000) # Give it some time to render (JAX compilation might take a bit)

    # 5. Take screenshot
    page.screenshot(path="/home/jules/verification/gui_screenshot.png", full_page=True)

if __name__ == "__main__":
    import os
    os.makedirs("/home/jules/verification", exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            verify_gui(page)
        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="/home/jules/verification/error_screenshot.png")
            raise e
        finally:
            browser.close()
