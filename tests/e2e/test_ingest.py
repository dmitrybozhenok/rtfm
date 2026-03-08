"""E2E tests for the ingestion UI flow."""

import pytest

pytestmark = pytest.mark.e2e


def test_ingest_panel_toggles(page):
    """Clicking Ingest button toggles the ingest panel."""
    page.goto("/")

    # Panel hidden initially
    assert not page.locator("#ingestPanel").is_visible()

    # Click to show
    page.click("button:has-text('Ingest')")
    page.wait_for_selector("#ingestPanel.show")
    assert page.locator("#ingestPanel").is_visible()

    # Click again to hide
    page.click("button:has-text('Ingest')")
    page.locator("#ingestPanel").wait_for(state="hidden")
    assert not page.locator("#ingestPanel").is_visible()


def test_ingest_panel_has_url_and_file_inputs(page):
    """Ingest panel has URL input and file upload."""
    page.goto("/")
    page.click("button:has-text('Ingest')")
    page.wait_for_selector("#ingestPanel.show")

    assert page.locator("#ingestUrl").is_visible()
    assert page.locator("#ingestFile").count() == 1
    assert page.locator("#btnIngestUrl").is_visible()
    assert page.locator("#btnIngestFile").is_visible()
    assert page.locator("#ingestRecursive").count() == 1


def test_ingest_url_empty_does_nothing(page):
    """Clicking Ingest URL with empty input does nothing."""
    page.goto("/")
    page.click("button:has-text('Ingest')")
    page.wait_for_selector("#ingestPanel.show")

    page.click("#btnIngestUrl")

    # No message should appear
    msg = page.text_content("#ingestMsg")
    assert msg == "" or msg.strip() == ""


def test_sources_panel_switches_from_ingest(page):
    """Opening Sources panel closes Ingest panel."""
    page.goto("/")

    page.click("button:has-text('Ingest')")
    page.wait_for_selector("#ingestPanel.show")

    page.click("button:has-text('Sources')")
    page.wait_for_selector("#sourcesPanel.show")

    # Ingest panel should be hidden
    assert not page.locator("#ingestPanel").is_visible()
