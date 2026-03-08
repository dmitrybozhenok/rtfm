"""E2E tests for the sources sidebar and viewer."""

import pytest

pytestmark = pytest.mark.e2e


def test_sources_sidebar_shows_documents(page):
    """Sources sidebar lists ingested documents."""
    page.goto("/")

    # Wait for sources to load
    page.wait_for_function(
        """document.querySelector('.source-item') !== null ||
           document.querySelector('#sourcesList')?.textContent?.includes('No sources')""",
        timeout=5000,
    )

    items = page.locator(".source-item")
    if items.count() > 0:
        # Should show source info
        assert page.locator(".source-icon").count() > 0
        assert page.locator(".source-meta").count() > 0

        # Total chunks text should be present
        total = page.text_content("#sourcesTotal")
        assert "chunks" in total


def test_source_click_opens_viewer(page):
    """Clicking a source opens the viewer drawer."""
    page.goto("/")

    # Wait for at least one source to load
    page.wait_for_selector(".source-item", timeout=5000)

    # Click first source
    page.locator(".source-item").first.click()

    # Viewer should open
    page.wait_for_selector(".viewer.open", timeout=10000)
    assert page.locator("#viewerTitle").text_content() != ""


def test_viewer_close_button(page):
    """Viewer can be closed with the X button."""
    page.goto("/")
    page.wait_for_selector(".source-item", timeout=5000)

    # Open viewer
    page.locator(".source-item").first.click()
    page.wait_for_selector(".viewer.open", timeout=10000)

    # Close it
    page.locator(".viewer-header button").click()
    page.locator(".viewer:not(.open)").wait_for(timeout=5000)


def test_sources_has_clear_button(page):
    """Sidebar has a Clear All button."""
    page.goto("/")
    clear_btn = page.locator("button", has_text="Clear All")
    assert clear_btn.is_visible()


def test_source_hover_shows_actions(page):
    """Hovering a source reveals action buttons (S, F, X)."""
    page.goto("/")
    page.wait_for_selector(".source-item", timeout=5000)

    # Hover to trigger action visibility
    page.locator(".source-item").first.hover()

    # Actions should contain X (delete)
    actions = page.locator(".source-item:first-child .source-actions button")
    assert actions.count() == 1
