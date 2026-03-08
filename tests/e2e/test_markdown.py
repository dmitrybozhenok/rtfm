"""E2E tests for markdown rendering in chat responses."""

import pytest

pytestmark = pytest.mark.e2e


def test_markdown_rendering_inline_code(page):
    """Inline code renders with <code> tags."""
    page.goto("/")

    # Inject a message with inline code to test rendering
    page.evaluate("""
        const chat = document.getElementById('chat');
        const welcome = document.querySelector('.welcome');
        if (welcome) welcome.remove();
        const div = document.createElement('div');
        div.className = 'msg assistant';
        div.innerHTML = '<div class="bubble">' + renderMarkdown('Use `git status` to check.') + '</div>';
        chat.appendChild(div);
    """)

    code = page.locator(".msg.assistant code")
    assert code.count() >= 1
    assert "git status" in code.first.text_content()


def test_markdown_rendering_code_block(page):
    """Code blocks render with <pre><code> tags."""
    page.goto("/")

    page.evaluate("""
        const chat = document.getElementById('chat');
        const welcome = document.querySelector('.welcome');
        if (welcome) welcome.remove();
        const div = document.createElement('div');
        div.className = 'msg assistant';
        div.innerHTML = '<div class="bubble">' + renderMarkdown('Example:\\n```\\ngit commit -m "hello"\\n```') + '</div>';
        chat.appendChild(div);
    """)

    pre = page.locator(".msg.assistant pre")
    assert pre.count() >= 1
    assert "git commit" in pre.first.text_content()


def test_markdown_rendering_bold(page):
    """Bold text renders with <strong> tags."""
    page.goto("/")

    page.evaluate("""
        const chat = document.getElementById('chat');
        const welcome = document.querySelector('.welcome');
        if (welcome) welcome.remove();
        const div = document.createElement('div');
        div.className = 'msg assistant';
        div.innerHTML = '<div class="bubble">' + renderMarkdown('This is **important** text.') + '</div>';
        chat.appendChild(div);
    """)

    strong = page.locator(".msg.assistant strong")
    assert strong.count() >= 1
    assert "important" in strong.first.text_content()


def test_markdown_rendering_italic(page):
    """Italic text renders with <em> tags."""
    page.goto("/")

    page.evaluate("""
        const chat = document.getElementById('chat');
        const welcome = document.querySelector('.welcome');
        if (welcome) welcome.remove();
        const div = document.createElement('div');
        div.className = 'msg assistant';
        div.innerHTML = '<div class="bubble">' + renderMarkdown('This is *emphasized* text.') + '</div>';
        chat.appendChild(div);
    """)

    em = page.locator(".msg.assistant em")
    assert em.count() >= 1
    assert "emphasized" in em.first.text_content()
