project = "Cqlib"
author = "Cqlib contributors"
copyright = "2026, Cqlib contributors"

from pygments.lexers.special import TextLexer
from sphinx.highlighting import lexers

extensions = [
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
language = "zh_CN"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

html_theme = "classic"
html_title = "Cqlib 文档"
html_show_sourcelink = False

myst_heading_anchors = 3

lexers["mermaid"] = TextLexer()
