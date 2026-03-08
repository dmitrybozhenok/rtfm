"""E2E tests for theme toggling and UI state."""

import pytest

pytestmark = pytest.mark.e2e


def test_theme_toggle(page):
    """Theme button switches between dark and light mode."""
    page.goto("/")

    # Default is dark (no data-theme attribute or empty)
    initial = page.evaluate("document.documentElement.getAttribute('data-theme')")

    page.click("button:has-text('Theme')")
    after_toggle = page.evaluate("document.documentElement.getAttribute('data-theme')")

    # Should have changed
    assert initial != after_toggle

    # Toggle back
    page.click("button:has-text('Theme')")
    back = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert back == initial


def test_theme_persists_across_reload(page):
    """Theme choice persists via localStorage after page reload."""
    page.goto("/")

    # Set to light
    page.evaluate("document.documentElement.setAttribute('data-theme', 'light')")
    page.evaluate("localStorage.setItem('rtfm_theme', 'light')")

    page.reload()

    theme = page.evaluate("document.documentElement.getAttribute('data-theme')")
    assert theme == "light"

    # Clean up
    page.evaluate("localStorage.removeItem('rtfm_theme')")


def test_health_dot_shows_status(page):
    """Health dot indicator shows green when server is healthy."""
    page.goto("/")

    # Wait for health check to complete
    page.wait_for_function(
        "document.getElementById('health').classList.contains('ok') || "
        "document.getElementById('health').classList.contains('degraded')",
        timeout=10000,
    )

    dot = page.locator("#health")
    classes = dot.get_attribute("class")
    # Should be ok or degraded (not bad, since server is running)
    assert "ok" in classes or "degraded" in classes


def test_session_id_in_localstorage(page):
    """Session ID is generated and stored in localStorage."""
    page.goto("/")

    session_id = page.evaluate("localStorage.getItem('rtfm_session')")
    assert session_id is not None
    assert len(session_id) > 10  # UUID-like
