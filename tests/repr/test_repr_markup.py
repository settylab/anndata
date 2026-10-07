"""Tests for the MarkupSafe trust boundary of the HTML repr."""

from __future__ import annotations

import numpy as np
import pytest
from markupsafe import Markup

from anndata import AnnData
from anndata._repr import (
    FormattedEntry,
    FormattedOutput,
    TypeFormatter,
    generate_repr_html,
    get_css,
    get_javascript,
    register_formatter,
    render_badge,
    render_copy_button,
    render_formatted_entry,
    render_header_badges,
    render_search_box,
    render_section,
    render_warning_icon,
)
from anndata._repr.components import (
    TypeCellConfig,
    render_category_list,
    render_entry_preview_cell,
    render_entry_type_cell,
    render_muted_span,
    render_name_cell,
    render_nested_content,
)
from anndata._repr.core import (
    render_empty_section,
    render_error_section,
    render_index_preview,
    render_truncation_indicator,
)
from anndata._repr.utils import escape_html, trusted_html

XSS = '<script>alert("x")</script>'


class _PlainStrHtmlType:
    """Marker type for formatters that return plain-str ``*_html`` fields."""


@pytest.fixture
def legacy_formatter():
    """A third-party formatter written before ``Markup`` existed."""

    @register_formatter
    class LegacyFormatter(TypeFormatter):
        sections = ("uns",)

        def can_format(self, obj, context):
            return isinstance(obj, _PlainStrHtmlType)

        def format(self, obj, context):
            return FormattedOutput(
                type_name="legacy",
                type_html='<b class="legacy-type">T</b>',
                preview_html='<span class="legacy-preview">hello <i>world</i></span>',
                expanded_html='<div class="legacy-expanded">more</div>',
            )

    return LegacyFormatter


def test_plain_str_html_fields_stay_trusted(legacy_formatter):
    """Backward compat: plain-str ``*_html`` from extensions is emitted verbatim."""
    adata = AnnData(np.zeros((3, 2), dtype="float32"))
    adata.uns["legacy"] = _PlainStrHtmlType()
    html = generate_repr_html(adata)
    assert '<span class="legacy-preview">hello <i>world</i></span>' in html
    assert '<div class="legacy-expanded">more</div>' in html
    assert "&lt;span" not in html
    # type_html is only displayed for mapping sections (appended below the type)
    mapping_row = render_formatted_entry(
        FormattedEntry(
            key="legacy", output=legacy_formatter().format(_PlainStrHtmlType(), None)
        ),
        append_type_html=True,
    )
    assert '<b class="legacy-type">T</b>' in mapping_row


def test_markup_html_fields_render_verbatim():
    out = FormattedOutput(
        type_name="t",
        preview_html=Markup("<span>{}</span>").format(XSS),
    )
    html = render_formatted_entry(FormattedEntry(key="k", output=out))
    assert "<span>&lt;script&gt;" in html
    assert XSS not in html


@pytest.mark.parametrize("field", ["preview", "type_name", "tooltip", "error"])
def test_plain_text_fields_are_escaped(field):
    kwargs = {"type_name": "t", field: XSS}
    html = render_formatted_entry(
        FormattedEntry(key="k", output=FormattedOutput(**kwargs))
    )
    assert XSS not in html
    assert "&lt;script&gt;" in html


def test_keys_are_escaped():
    html = render_formatted_entry(
        FormattedEntry(key=XSS, output=FormattedOutput(type_name="t"))
    )
    assert XSS not in html
    assert "&lt;script&gt;" in html


def test_user_data_in_full_repr_is_escaped():
    adata = AnnData(np.zeros((2, 2), dtype="float32"))
    adata.obs[XSS] = [1, 2]
    adata.uns[XSS] = XSS
    adata.obsm[XSS] = np.zeros((2, 3))
    html = generate_repr_html(adata)
    assert XSS not in html
    assert "&lt;script&gt;" in html


def test_render_helpers_return_markup():
    entry = FormattedEntry(key="k", output=FormattedOutput(type_name="t", preview="p"))
    results = {
        "render_badge": render_badge("a", "b", "c"),
        "render_copy_button": render_copy_button("a"),
        "render_header_badges": render_header_badges(is_view=True),
        "render_search_box": render_search_box("id"),
        "render_warning_icon": render_warning_icon(["w"]),
        "render_muted_span": render_muted_span("a"),
        "render_name_cell": render_name_cell("a"),
        "render_nested_content": render_nested_content("<p>x</p>"),
        "render_category_list": render_category_list(["a"], ["red"], 5),
        "render_entry_type_cell": render_entry_type_cell(
            TypeCellConfig(type_name="t", css_class="c")
        ),
        "render_entry_preview_cell": render_entry_preview_cell(preview_text="p"),
        "render_section": render_section("s", "<div>x</div>", n_items=1),
        "render_empty_section": render_empty_section("s"),
        "render_error_section": render_error_section("s", "e"),
        "render_index_preview": render_index_preview(AnnData(np.zeros((2, 2)))),
        "render_truncation_indicator": render_truncation_indicator(3),
        "render_formatted_entry": render_formatted_entry(entry),
        "generate_repr_html": generate_repr_html(AnnData(np.zeros((2, 2)))),
        "get_css": get_css(),
        "get_javascript": get_javascript("c"),
        "escape_html": escape_html(XSS),
    }
    not_markup = {k: type(v) for k, v in results.items() if not isinstance(v, Markup)}
    assert not not_markup


def test_render_helpers_accept_plain_str_html():
    """Public helpers keep trusting plain-str HTML arguments."""
    section = render_section("s", '<div class="mine">x</div>', n_items=1)
    assert '<div class="mine">x</div>' in section
    nested = render_nested_content('<p class="mine">x</p>')
    assert '<p class="mine">x</p>' in nested
    cell = render_entry_preview_cell(preview_html='<b class="mine">x</b>')
    assert '<b class="mine">x</b>' in cell


def test_trusted_html_is_idempotent_and_wraps_str():
    m = Markup("<b>x</b>")
    assert trusted_html(m) is m
    wrapped = trusted_html("<b>x</b>")
    assert isinstance(wrapped, Markup)
    assert wrapped == "<b>x</b>"


def test_escape_html_replaces_null_bytes_and_is_not_double_escaped():
    out = escape_html("a\x00<b>")
    assert out == "a�&lt;b&gt;"
    assert Markup("{}").format(out) == out


def test_join_markup_matches_markup_join():
    from anndata._repr.utils import join_markup

    parts = [Markup("<b>"), escape_html("a<b"), Markup("</b>")]
    assert join_markup(parts) == Markup("").join(parts)
    assert join_markup(parts, ", ") == Markup(", ").join(parts)
    assert isinstance(join_markup(parts), Markup)


def test_markup_plus_plain_str_escapes_the_plain_side():
    """Documented caveat: compose with Markup, not with plain-str ``+``."""
    assert render_badge("x") + "<hr>" == render_badge("x") + Markup("&lt;hr&gt;")
