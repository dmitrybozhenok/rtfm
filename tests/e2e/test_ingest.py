"""E2E tests for the ingestion UI flow."""

import pytest

pytestmark = pytest.mark.e2e


def test_add_source_area_visible(page):
    """Add source area is visible in sidebar."""
    page.goto("/")
    assert page.locator("#ingestFile").count() == 1
    assert page.locator("#btnIngestFile").is_visible()
