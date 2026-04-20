"""
Jinja2 Environment for the AnnData HTML repr (middle-ground POC).

This module wires Jinja2 into the existing repr pipeline in a minimal way:

- A single autoescape-enabled ``Environment`` loads templates from
  ``anndata._repr.templates``.
- The existing formatter machinery still produces HTML fragments as strings;
  the top-level renderer wraps those fragments in ``markupsafe.Markup`` at the
  boundary so they pass through autoescape verbatim.
- Any additional values injected directly into the outer template (container
  id, depth, inline style, etc.) are autoescaped by default, which closes the
  "forgot to call ``html.escape()``" class of bug for those specific
  insertions.

This is deliberately narrow in scope. It illustrates the trust contract
(``Markup`` = trusted, ``str`` = untrusted) without rewriting the per-type
formatters.
"""

from __future__ import annotations

from functools import cache

from jinja2 import Environment, PackageLoader, select_autoescape


@cache
def get_env() -> Environment:
    return Environment(
        loader=PackageLoader("anndata._repr", "templates"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
