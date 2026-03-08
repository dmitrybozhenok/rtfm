"""E2E tests for the sources panel."""

import pytest

pytestmark = pytest.mark.e2e


def test_sources_panel_toggles(page):
    """Clicking Sources button toggles the sources panel."""
    page.goto("/")

    assert not page.locator("#sourcesPanel").is_visible()

    page.click("button:has-text('Sources')")
    page.wait_for_selector("#sourcesPanel.show")
    assert page.locator("#sourcesPanel").is_visible()

    page.click("button:has-text('Sources')")
    page.locator("#sourcesPanel").wait_for(state="hidden")


def test_sources_panel_shows_documents(page):
    """Sources panel lists ingested documents with type badges."""
    page.goto("/")
    page.click("button:has-text('Sources')")
    page.wait_for_selector("#sourcesPanel.show")

    # Wait for sources to load (either items or empty message)
    page.wait_for_function(
        """document.querySelector('.source-item') !== null ||
           document.querySelector('.sources-empty')?.textContent?.includes('No documents')""",
        timeout=5000,
    )

    items = page.locator(".source-item")
    if items.count() > 0:
        # Should have type badges
        badges = page.locator(".source-badge")
        assert badges.count() > 0

        # Total text should be present
        total = page.text_content("#sourcesTotal")
        assert "sources" in total
        assert "chunks" in total


def test_sources_has_clear_button(page):
    """Sources panel has a Clear All Documents button."""
    page.goto("/")
    page.click("button:has-text('Sources')")
    page.wait_for_selector("#sourcesPanel.show")

    clear_btn = page.locator("button:has-text('Clear All Documents')")
    assert clear_btn.is_visible()
