"""E2E tests for the chat flow."""

import pytest

pytestmark = pytest.mark.e2e


def test_page_loads_with_welcome(page):
    """Page loads at / and shows the welcome message."""
    page.goto("/")
    page.wait_for_selector(".welcome")
    assert page.text_content(".welcome h2") == "Ask me anything about your docs"


def test_header_elements_present(page):
    """Header shows title and control buttons."""
    page.goto("/")
    assert "RTFM" in page.text_content("header h1")
    assert page.locator("button", has_text="Theme").is_visible()
    assert page.locator("#health").is_visible()


def test_sidebar_visible_by_default(page):
    """Sources sidebar is visible on desktop."""
    page.goto("/")
    sidebar = page.locator("#sidebar")
    assert sidebar.is_visible()
    assert page.locator(".sidebar-header").is_visible()


def test_send_question_gets_response(page):
    """Type a question and get a streaming assistant response."""
    page.goto("/")

    page.fill("#question", "What is Git?")
    page.click("#send")

    # Welcome should disappear
    page.wait_for_selector(".welcome", state="detached", timeout=5000)

    # User message appears
    user_msg = page.locator(".msg.user .bubble")
    assert user_msg.count() >= 1
    assert "What is Git?" in user_msg.first.text_content()

    # Assistant response appears (wait for streaming to produce content)
    assistant_bubble = page.locator(".msg.assistant .bubble")
    assistant_bubble.first.wait_for(timeout=60000)

    # Wait for response content (not empty)
    page.wait_for_function(
        "document.querySelector('.msg.assistant .bubble')?.textContent?.length > 5",
        timeout=60000,
    )
    text = assistant_bubble.first.text_content()
    assert len(text) > 5


def test_empty_input_no_request(page):
    """Empty input should not send a request."""
    page.goto("/")

    page.click("#send")

    # Welcome should still be there
    assert page.locator(".welcome").is_visible()
    assert page.locator(".msg").count() == 0


def test_enter_sends_question(page):
    """Pressing Enter (without Shift) sends the question."""
    page.goto("/")

    page.fill("#question", "What is a branch?")
    page.press("#question", "Enter")

    # User message should appear
    page.wait_for_selector(".msg.user", timeout=5000)
    assert "What is a branch?" in page.locator(".msg.user .bubble").first.text_content()


def test_shift_enter_adds_newline(page):
    """Shift+Enter should add a newline, not send."""
    page.goto("/")

    page.fill("#question", "line one")
    page.press("#question", "Shift+Enter")
    page.type("#question", "line two")

    value = page.input_value("#question")
    assert "line one" in value
    assert "line two" in value
    assert page.locator(".msg").count() == 0


def test_send_button_disabled_during_streaming(page):
    """Send button is disabled while waiting for a response."""
    page.goto("/")

    page.fill("#question", "What is Git?")
    page.click("#send")

    page.wait_for_selector("#send[disabled]", timeout=5000)
    page.wait_for_selector("#send:not([disabled])", timeout=60000)
