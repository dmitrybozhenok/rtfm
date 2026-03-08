"""E2E tests for the ingestion UI flow."""

import pytest

pytestmark = pytest.mark.e2e


def test_add_source_area_visible(page):
    """Add source area is visible in sidebar."""
    page.goto("/")
    assert page.locator("#ingestUrl").is_visible()
    assert page.locator("#btnIngestUrl").is_visible()
    assert page.locator("#ingestFile").count() == 1
    assert page.locator("#btnIngestFile").is_visible()


def test_ingest_url_empty_does_nothing(page):
    """Clicking Add with empty URL input does nothing."""
    page.goto("/")
    page.click("#btnIngestUrl")

    # No message should appear
    msg = page.text_content("#ingestMsg")
    assert msg == "" or msg.strip() == ""


def test_recursive_checkbox_present(page):
    """Recursive checkbox is present in add source area."""
    page.goto("/")
    assert page.locator("#ingestRecursive").count() == 1
