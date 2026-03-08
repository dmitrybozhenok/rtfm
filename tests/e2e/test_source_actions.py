"""E2E tests for source-level actions (delete)."""

import pytest

pytestmark = pytest.mark.e2e


def _get_first_source(page) -> str | None:
    """Wait for sources to load and return the first source name."""
    page.goto("/")
    try:
        page.wait_for_selector(".source-item", timeout=5000)
    except Exception:
        pytest.skip("No sources ingested")
    return page.evaluate(
        "document.querySelector('.source-item')?.getAttribute('title')"
    )


def test_delete_button_present(page):
    """Delete button (X) is visible on source hover."""
    source = _get_first_source(page)
    if not source:
        pytest.skip("No sources")

    page.locator(".source-item").first.hover()
    actions = page.locator(".source-item:first-child .source-actions button")
    assert actions.count() == 1
    assert actions.first.get_attribute("title") == "Delete"
