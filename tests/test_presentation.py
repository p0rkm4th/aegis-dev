from aegis.presentation import render_safe_markdown


def test_safe_markdown_renders_common_assistant_formatting() -> None:
    rendered = render_safe_markdown(
        "## Result\n\n- one\n- two\n\n```python\nprint(1)\n```\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |"
    )

    assert "<h2>Result</h2>" in rendered
    assert "<li>one</li>" in rendered
    assert '<pre><code class="language-python">print(1)' in rendered
    assert "<table>" in rendered


def test_safe_markdown_escapes_html_and_rejects_dangerous_links() -> None:
    rendered = render_safe_markdown(
        '<script>alert("no")</script> [bad](javascript:alert(1)) [good](https://example.com)'
    )

    assert "<script>" not in rendered
    assert "javascript:" not in rendered
    assert 'href="https://example.com"' in rendered
    assert 'target="_blank"' in rendered
