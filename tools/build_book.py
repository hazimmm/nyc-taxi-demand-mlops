r"""Build the study book in ``docs/learning/`` into one PDF textbook.

Run from anywhere::

    .venv\Scripts\python.exe tools\build_book.py

The pipeline has four steps:

1. Discover the Markdown files (cover, numbered chapters, appendices).
2. Convert them to one HTML document (markdown-it-py, Pygments, ``book.css``).
3. Render the Mermaid diagrams to SVG in headless Chrome or Edge.
4. Print the HTML to PDF with the same browser.

The Markdown stays the source of truth: this script never changes it. Intermediate
files go to ``build/book/`` and the pinned copy of mermaid.js to ``build/cache/``.

Environment variables:

* ``BOOK_BROWSER`` - path of the Chrome, Chromium or Edge executable to use.
* ``BOOK_BROWSER_FLAGS`` - extra browser flags, separated by spaces
  (for example ``--no-sandbox`` inside a container).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, unquote

from markdown_it import MarkdownIt
from markdown_it.token import Token
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name
from pygments.util import ClassNotFound

log = logging.getLogger("build_book")

# ---------------------------------------------------------------------------
# Locations (all resolved from this file, never from the current directory)
# ---------------------------------------------------------------------------

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
BOOK_DIR = REPO_ROOT / "docs" / "learning"
OUTPUT_PDF = BOOK_DIR / "nyc-taxi-demand-mlops-textbook.pdf"
BUILD_DIR = REPO_ROOT / "build" / "book"
CACHE_DIR = REPO_ROOT / "build" / "cache"
CSS_FILE = TOOLS_DIR / "book.css"
DIAGRAM_SCRIPT = TOOLS_DIR / "book_diagrams.js"

COVER_FILE = "README.md"
COVER_TITLE = "About this book"
REPOSITORY_NAME = "nyc-taxi-demand-mlops"
GITHUB_BLOB_URL = f"https://github.com/hazimmm/{REPOSITORY_NAME}/blob/dev/"

# ---------------------------------------------------------------------------
# Mermaid: one pinned version, fetched once, checked against a known hash
# ---------------------------------------------------------------------------

MERMAID_VERSION = "11.12.0"
MERMAID_URL = f"https://cdn.jsdelivr.net/npm/mermaid@{MERMAID_VERSION}/dist/mermaid.min.js"
MERMAID_SHA256 = "07e37dfa97b337ccc85365d57eddf99b9706f09db3b59b260d0333b23b343c4b"

# ---------------------------------------------------------------------------
# Page geometry (A4). The CSS @page rules are generated from these values.
# ---------------------------------------------------------------------------

PAGE_WIDTH_MM = 210.0
MARGIN_TOP_MM = 24.0
MARGIN_SIDE_MM = 20.0
MARGIN_BOTTOM_MM = 24.0
PX_PER_MM = 96.0 / 25.4
TEXT_WIDTH_PX = (PAGE_WIDTH_MM - 2 * MARGIN_SIDE_MM) * PX_PER_MM
MAX_DIAGRAM_HEIGHT_PX = 150.0 * PX_PER_MM
DIAGRAM_FONT_PT = 10.5  # 14 px, set in book_diagrams.js
DIAGRAM_BASE_SCALE = 0.86  # labels print at 9 pt, a little smaller than the body text
MIN_LABEL_PT = 6.5  # below this a diagram is reported as hard to read

SHORT_CODE_LINES = 20  # code blocks up to this length are not split across pages
WIDE_TABLE_COLUMNS = 5
MIN_BROWSER_MAJOR = 131  # first Chromium with CSS page-margin boxes (page numbers)
BROWSER_TIMEOUT_S = 180
DIAGRAM_TIME_BUDGET_MS = 30_000
TOC_PAGE_PLACEHOLDER = "000"

BROWSER_ENV = "BOOK_BROWSER"
BROWSER_FLAGS_ENV = "BOOK_BROWSER_FLAGS"
WINDOWS_BROWSERS = (
    Path("Google") / "Chrome" / "Application" / "chrome.exe",
    Path("Microsoft") / "Edge" / "Application" / "msedge.exe",
)
WINDOWS_ROOT_VARS = ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
MAC_BROWSERS = (
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
)
PATH_BROWSERS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "microsoft-edge",
    "msedge",
)

ALERT_RE = re.compile(r"\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]")
CAPTION_RE = re.compile(r"(Figure|Table)\s+\S+")
CHAPTER_FILE_RE = re.compile(r"(\d+)-.+\.md")
APPENDIX_FILE_RE = re.compile(r"appendix-([a-z0-9]+)-.+\.md")
NAV_LABEL_RE = re.compile(r"(Previous|Next)\b|Contents$")
SCHEME_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*:|//")
TITLE_SPLIT_RE = re.compile(r"(.{1,24}?)(\s+[\u2013\u2014]\s+)(.+)")
DIAGRAM_PLACEHOLDER = "<!--book-diagram-{index}-->"
REPORT_RE = re.compile(r'<pre id="book-diagram-report">(.*?)</pre>', re.DOTALL)


class BuildError(Exception):
    """A problem that stops the build, with a message written for the reader."""


@dataclass(frozen=True)
class Source:
    """One Markdown file of the book.

    Attributes:
        path: Absolute path of the file.
        kind: ``"cover"``, ``"chapter"`` or ``"appendix"``.
        anchor: HTML id of the section built from this file.
    """

    path: Path
    kind: str
    anchor: str


@dataclass
class Diagram:
    """One Mermaid diagram found in the Markdown.

    Attributes:
        index: Position of the diagram in the whole book, from 0.
        source: File that contains the diagram.
        number: Position of the diagram inside its file, from 1.
        line: Line of the opening code fence in the file.
        code: Mermaid source text.
    """

    index: int
    source: Source
    number: int
    line: int
    code: str

    @property
    def location(self) -> str:
        """Where to find the diagram, for messages."""
        relative = self.source.path.relative_to(REPO_ROOT).as_posix()
        return f"{relative}, diagram {self.number} (line {self.line})"


@dataclass
class Section:
    """One file after parsing: a part of the front matter, a chapter or an appendix.

    Attributes:
        source: The file the section comes from.
        title: Text of the first level-1 heading.
        tokens: markdown-it tokens of the body, without the title heading.
        body: Rendered HTML of the body.
    """

    source: Source
    title: str
    tokens: list[Token]
    body: str = ""


@dataclass
class Book:
    """Everything known about the book while it is being built.

    Attributes:
        sections: All sections in reading order; the cover file is first.
        by_path: Lookup from a Markdown file to its source entry.
        ids: Every HTML id a link may point to.
        diagrams: All Mermaid diagrams, in reading order.
        parts: Part name for each chapter file, as listed in the cover file.
        groups: Contents of the table of contents: part names with their sections.
        warnings: Problems that do not stop the build.
    """

    sections: list[Section] = field(default_factory=list)
    by_path: dict[Path, Source] = field(default_factory=dict)
    ids: set[str] = field(default_factory=set)
    diagrams: list[Diagram] = field(default_factory=list)
    parts: dict[Path, str] = field(default_factory=dict)
    groups: list[tuple[str, list[Section]]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        """Record and log a problem that does not stop the build."""
        self.warnings.append(message)
        log.warning("warning: %s", message)


@dataclass(frozen=True)
class PdfLayout:
    """Facts read back from a printed PDF.

    Attributes:
        page_count: Number of pages.
        pages: Page number (from 1) on which each section anchor starts.
        outline_entries: Number of top-level bookmarks.
    """

    page_count: int
    pages: dict[str, int]
    outline_entries: int


# ---------------------------------------------------------------------------
# Step 1 - discover the files
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Turn heading text into an anchor the way GitHub does.

    Args:
        text: Plain text of a heading.

    Returns:
        Lower-case text with punctuation removed and spaces replaced by hyphens.
    """
    kept = re.sub(r"[^\w\- ]", "", text.strip().lower())
    return kept.replace(" ", "-")


def discover_sources(book_dir: Path) -> list[Source]:
    """Find the book's Markdown files and put them in reading order.

    The order is: the cover file, the numbered chapters by number, then the
    appendices by letter. Nothing is hard-coded, so a new chapter file is
    picked up automatically.

    Args:
        book_dir: Folder that holds the Markdown files.

    Returns:
        The sources in reading order, cover first.

    Raises:
        BuildError: If the folder or the cover file is missing, or there are no chapters.
    """
    if not book_dir.is_dir():
        raise BuildError(f"Book folder not found: {book_dir}")
    cover = book_dir / COVER_FILE
    if not cover.is_file():
        raise BuildError(f"Cover file not found: {cover}")

    chapters: list[tuple[int, str, Path]] = []
    appendices: list[tuple[str, Path]] = []
    for path in sorted(book_dir.glob("*.md")):
        if path.name == COVER_FILE:
            continue
        if match := CHAPTER_FILE_RE.fullmatch(path.name):
            chapters.append((int(match.group(1)), path.name, path))
        elif match := APPENDIX_FILE_RE.fullmatch(path.name):
            appendices.append((match.group(1), path))
        else:
            log.warning(
                "warning: %s is not a chapter (NN-name.md) or an appendix "
                "(appendix-x-name.md); it is left out of the book",
                path.name,
            )
    if not chapters:
        raise BuildError(f"No chapter files (NN-name.md) found in {book_dir}")

    sources = [Source(cover.resolve(), "cover", slugify(COVER_TITLE))]
    sources += [Source(p.resolve(), "chapter", slugify(p.stem)) for _, _, p in sorted(chapters)]
    sources += [Source(p.resolve(), "appendix", slugify(p.stem)) for _, p in sorted(appendices)]
    return sources


# ---------------------------------------------------------------------------
# Step 2 - Markdown to HTML
# ---------------------------------------------------------------------------


def highlight_code(code: str, language: str, _attrs: str) -> str:
    """Render one fenced code block with syntax colours.

    Args:
        code: Text inside the fence.
        language: Language named after the opening fence; may be empty.
        _attrs: Rest of the fence line; not used.

    Returns:
        A complete ``<pre>`` element. Short blocks get the class ``short`` so
        that the stylesheet keeps them on one page.
    """
    try:
        lexer = get_lexer_by_name(language) if language else TextLexer()
    except ClassNotFound:
        lexer = TextLexer()
    body = highlight(code, lexer, HtmlFormatter(nowrap=True)).rstrip("\n")
    classes = "code short" if code.count("\n") <= SHORT_CODE_LINES else "code"
    label = "" if language in ("", "text") else f' data-lang="{html.escape(language)}"'
    return f'<pre class="{classes}"{label}><code class="highlight">{body}</code></pre>\n'


def create_markdown_parser() -> MarkdownIt:
    """Create the parser: CommonMark with GitHub tables, raw HTML and highlighting."""
    parser = MarkdownIt("commonmark", {"html": True, "highlight": highlight_code})
    return parser.enable(["table", "strikethrough"])


def html_token(content: str) -> Token:
    """Make a token that the renderer writes out unchanged."""
    return Token("html_block", "", 0, content=content)


def plain_text(inline: Token) -> str:
    """Return the visible text of an inline token, without any markup."""
    children = inline.children or []
    return "".join(child.content for child in children if child.type in ("text", "code_inline"))


def split_link(href: str, source: Source) -> tuple[Path | None, str]:
    """Split a relative link into the file it points to and its anchor.

    Args:
        href: Link target as written in the Markdown; must not be an external link.
        source: File that contains the link.

    Returns:
        The absolute path of the target (``None`` for a link inside the same
        file) and the anchor without ``#`` (empty if there is none).
    """
    path_part, _, fragment = href.partition("#")
    if not path_part:
        return None, fragment
    return (source.path.parent / unquote(path_part)).resolve(), fragment


def book_target(href: str, source: Source, book: Book) -> Source | None:
    """Return the book file a link points to, or ``None`` if it leaves the book."""
    if SCHEME_RE.match(href):
        return None
    path, _ = split_link(href, source)
    return source if path is None else book.by_path.get(path)


def take_title(tokens: list[Token], source: Source, book: Book) -> tuple[str, list[Token]]:
    """Remove the first level-1 heading from the tokens and return its text.

    Args:
        tokens: Tokens of one file.
        source: The file, for the fallback title and the warning.
        book: The book, to record a warning.

    Returns:
        The title and the remaining tokens.
    """
    for i, token in enumerate(tokens):
        if token.type == "heading_open" and token.tag == "h1":
            return plain_text(tokens[i + 1]), tokens[:i] + tokens[i + 3 :]
    book.warn(f"{source.path.name} has no '# ' title; the file name is used instead")
    return source.path.stem.replace("-", " ").capitalize(), tokens


def assign_heading_ids(section: Section, book: Book) -> None:
    """Give every heading a unique id of the form ``<section>--<github-slug>``.

    Args:
        section: Section whose heading tokens are changed in place.
        book: The book; the new ids are added to ``book.ids``.
    """
    anchor = section.source.anchor
    book.ids.add(anchor)
    seen: dict[str, int] = {}
    for i, token in enumerate(section.tokens):
        if token.type != "heading_open":
            continue
        slug = slugify(plain_text(section.tokens[i + 1]))
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        heading_id = f"{anchor}--{slug}" if count == 0 else f"{anchor}--{slug}-{count}"
        token.attrSet("id", heading_id)
        book.ids.add(heading_id)


def collect_parts(section: Section, book: Book) -> None:
    """Read from the cover file which part each chapter belongs to.

    A chapter belongs to the nearest heading above the table or list that
    links to it, for example ``### Part I - The problem``.

    Args:
        section: The cover section, before its links are rewritten.
        book: The book; the result is stored in ``book.parts``.
    """
    heading = ""
    depth = 0
    tokens = section.tokens
    for i, token in enumerate(tokens):
        if token.type == "heading_open":
            heading = plain_text(tokens[i + 1])
        elif token.type in ("table_open", "bullet_list_open", "ordered_list_open"):
            depth += 1
        elif token.type in ("table_close", "bullet_list_close", "ordered_list_close"):
            depth -= 1
        elif token.type == "inline" and depth > 0:
            for child in token.children or []:
                if child.type != "link_open":
                    continue
                target = book_target(str(child.attrGet("href") or ""), section.source, book)
                if target is not None and target.kind != "cover":
                    book.parts.setdefault(target.path, heading)


def is_navigation(inline: Token, source: Source, book: Book) -> bool:
    """Tell whether a paragraph is the "Previous | Contents | Next" line.

    Args:
        inline: Inline token of the paragraph.
        source: File that contains the paragraph.
        book: The book, to resolve the links.

    Returns:
        True if the paragraph holds only links to book files separated by ``|``
        and at least one of them is labelled Previous, Contents or Next.
    """
    links = 0
    depth = 0
    labels: list[str] = []
    for child in inline.children or []:
        if child.type == "link_open":
            if book_target(str(child.attrGet("href") or ""), source, book) is None:
                return False
            links += 1
            depth += 1
        elif child.type == "link_close":
            depth -= 1
        elif depth > 0:
            labels.append(child.content)
        elif child.content.strip() not in ("", "|"):
            return False
    return links > 0 and any(NAV_LABEL_RE.match(label) for label in labels)


def remove_navigation(section: Section, book: Book) -> None:
    """Drop the navigation line, and the rule above it, from a section."""
    kept: list[Token] = []
    tokens = section.tokens
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.type == "paragraph_open" and is_navigation(tokens[i + 1], section.source, book):
            if kept and kept[-1].type == "hr":
                kept.pop()
            i += 3
            continue
        kept.append(token)
        i += 1
    section.tokens = kept


def alert_kind(tokens: list[Token], i: int) -> str | None:
    """Return the alert type if the block quote at ``i`` is a GitHub alert.

    Args:
        tokens: Tokens of one file.
        i: Index of a token.

    Returns:
        ``"NOTE"``, ``"TIP"``, ``"IMPORTANT"``, ``"WARNING"`` or ``"CAUTION"``,
        or ``None`` if the token does not open an alert.
    """
    if tokens[i].type != "blockquote_open" or tokens[i + 1].type != "paragraph_open":
        return None
    children = tokens[i + 2].children or []
    if not children or children[0].type != "text":
        return None
    match = ALERT_RE.fullmatch(children[0].content.strip())
    marker_alone = len(children) == 1 or children[1].type in ("softbreak", "hardbreak")
    return match.group(1) if match and marker_alone else None


def mark_alerts(section: Section) -> None:
    """Turn ``> [!NOTE]`` block quotes into callout boxes with a label."""
    kept: list[Token] = []
    tokens = section.tokens
    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = alert_kind(tokens, i)
        if kind is None:
            kept.append(token)
            i += 1
            continue
        token.tag = "div"
        token.attrSet("class", f"callout callout-{kind.lower()}")
        closing = next(
            t for t in tokens[i:] if t.type == "blockquote_close" and t.level == token.level
        )
        closing.tag = "div"
        kept.append(token)
        kept.append(html_token(f'<p class="callout-title">{kind.capitalize()}</p>\n'))
        inline = tokens[i + 2]
        inline.children = (inline.children or [])[2:]  # the marker and its line break
        if inline.children:
            kept.extend(tokens[i + 1 : i + 4])
        i += 4
    section.tokens = kept


def caption_at(tokens: list[Token], i: int) -> bool:
    """Tell whether a caption paragraph starts at index ``i``.

    A caption is a paragraph written entirely in italics that starts with
    "Figure" or "Table" and a number, such as ``*Figure 4.1 - The layers.*``.
    """
    if i + 2 >= len(tokens) or tokens[i].type != "paragraph_open":
        return False
    children = tokens[i + 1].children or []
    if len(children) < 3 or children[0].type != "em_open" or children[-1].type != "em_close":
        return False
    depth = 0
    for position, child in enumerate(children):
        depth += {"em_open": 1, "em_close": -1}.get(child.type, 0)
        if depth == 0 and position < len(children) - 1:
            return False  # the italics end before the paragraph does
    return bool(CAPTION_RE.match(children[1].content))


def mark_figures(section: Section, book: Book) -> None:
    """Wrap diagrams, captioned listings and table captions for the print layout.

    * A ``mermaid`` fence becomes a ``<figure>`` with a placeholder that is later
      replaced by the rendered SVG. The diagram source is added to ``book.diagrams``.
    * A caption directly below a code block or diagram becomes its ``<figcaption>``,
      so the two cannot be separated by a page break.
    * A caption directly below a table gets the class ``caption``.

    Args:
        section: Section whose tokens are rewritten.
        book: The book, to collect the diagrams.
    """
    kept: list[Token] = []
    tokens = section.tokens
    diagrams_in_file = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        has_caption = caption_at(tokens, i + 1)
        is_diagram = token.type == "fence" and token.info.strip().split()[:1] == ["mermaid"]
        if token.type == "table_close" and has_caption:
            tokens[i + 1].attrSet("class", "caption")
        if token.type != "fence" or not (is_diagram or has_caption):
            kept.append(token)
            i += 1
            continue

        if is_diagram:
            diagrams_in_file += 1
            line = (token.map[0] if token.map else 0) + 1
            diagram = Diagram(
                len(book.diagrams), section.source, diagrams_in_file, line, token.content
            )
            book.diagrams.append(diagram)
            placeholder = DIAGRAM_PLACEHOLDER.format(index=diagram.index)
            kept.append(html_token(f'<figure class="diagram">\n{placeholder}\n'))
        else:
            kept.append(html_token('<figure class="listing">\n'))
            kept.append(token)
        i += 1
        if has_caption:
            tokens[i].tag = tokens[i + 2].tag = "figcaption"
            kept.extend(tokens[i : i + 3])
            i += 3
        kept.append(html_token("</figure>\n"))
    section.tokens = kept


def mark_tables(section: Section) -> None:
    """Add the class ``wide`` to tables with many columns (smaller type in print)."""
    tokens = section.tokens
    for i, token in enumerate(tokens):
        if token.type != "table_open":
            continue
        columns = 0
        for later in tokens[i:]:
            if later.type == "thead_close":
                break
            columns += later.type == "th_open"
        if columns >= WIDE_TABLE_COLUMNS:
            token.attrSet("class", "wide")


def rewrite_href(href: str, source: Source, book: Book) -> str:
    """Translate one Markdown link target into what the PDF needs.

    * External links are kept.
    * Links to another book file become internal links to that chapter, or to
      the heading inside it when the link has an anchor.
    * Links to other files of the repository point at the file on GitHub.

    Args:
        href: Link target as written in the Markdown.
        source: File that contains the link.
        book: The book, to look up files and heading ids.

    Returns:
        The new link target.
    """
    if SCHEME_RE.match(href):
        return href
    where = source.path.name
    path, fragment = split_link(href, source)
    target = source if path is None else book.by_path.get(path)

    if target is not None:
        if not fragment:
            return f"#{target.anchor}"
        heading_id = f"{target.anchor}--{fragment.lower()}"
        if heading_id in book.ids:
            return f"#{heading_id}"
        book.warn(f"{where}: link '{href}' names a heading that does not exist")
        return f"#{target.anchor}"

    assert path is not None  # a same-file link always has a target
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        book.warn(f"{where}: link '{href}' points outside the repository; left unchanged")
        return href
    if not path.exists():
        book.warn(f"{where}: link '{href}' points to '{relative}', which does not exist")
    suffix = f"#{fragment}" if fragment else ""
    return GITHUB_BLOB_URL + quote(relative) + suffix


def rewrite_links(section: Section, book: Book) -> None:
    """Rewrite every link, and make local image paths absolute, in a section."""
    for token in section.tokens:
        for child in token.children or []:
            if child.type == "link_open":
                href = str(child.attrGet("href") or "")
                child.attrSet("href", rewrite_href(href, section.source, book))
            elif child.type == "image":
                src = str(child.attrGet("src") or "")
                if not SCHEME_RE.match(src):
                    child.attrSet("src", (section.source.path.parent / unquote(src)).as_uri())


def expand_details(body: str) -> str:
    """Replace ``<details>`` blocks with always-open answer boxes.

    A PDF cannot be clicked open, so the hidden answers must be printed.
    Code samples are not affected: their ``<`` is already escaped.

    Args:
        body: Rendered HTML of one section.

    Returns:
        The HTML with every ``<details>`` turned into ``<div class="answer">``.
    """
    body = re.sub(r"<details[^>]*>", '<div class="answer">', body)
    body = re.sub(
        r"<summary[^>]*>(.*?)</summary>", r'<p class="answer-title">\1</p>', body, flags=re.DOTALL
    )
    return body.replace("</details>", "</div>")


def load_book(sources: list[Source]) -> Book:
    """Parse all Markdown files and prepare them for printing.

    Args:
        sources: The files in reading order.

    Returns:
        The book with every section rendered to HTML. Diagrams are still placeholders.
    """
    parser = create_markdown_parser()
    book = Book(by_path={source.path: source for source in sources})

    # First pass: every heading id must be known before any link is rewritten.
    for source in sources:
        tokens = parser.parse(source.path.read_text(encoding="utf-8"))
        title, tokens = take_title(tokens, source, book)
        section = Section(source, title, tokens)
        assign_heading_ids(section, book)
        book.sections.append(section)

    collect_parts(book.sections[0], book)
    book.groups = toc_groups(book)

    for section in book.sections:
        remove_navigation(section, book)
        mark_alerts(section)
        mark_figures(section, book)
        mark_tables(section)
        rewrite_links(section, book)
        rendered = parser.renderer.render(section.tokens, parser.options, {})
        section.body = expand_details(rendered)
    return book


# ---------------------------------------------------------------------------
# Step 3 - the browser and the diagrams
# ---------------------------------------------------------------------------


def browser_candidates() -> list[Path]:
    """List the usual install locations of Chrome and Edge on this machine."""
    candidates: list[Path] = []
    for variable in WINDOWS_ROOT_VARS:
        if root := os.environ.get(variable):
            candidates += [Path(root) / relative for relative in WINDOWS_BROWSERS]
    candidates += MAC_BROWSERS
    for name in PATH_BROWSERS:
        if found := shutil.which(name):
            candidates.append(Path(found))
    return candidates


def find_browser() -> Path:
    """Find a Chromium-based browser that can print to PDF.

    The search order is: the ``BOOK_BROWSER`` environment variable, the usual
    Chrome and Edge folders on Windows, the usual applications on macOS, then
    the command names used on Linux.

    Returns:
        Path of the browser executable.

    Raises:
        BuildError: If the override does not exist or no browser is found.
    """
    if override := os.environ.get(BROWSER_ENV):
        path = Path(override)
        if path.is_file():
            return path
        if found := shutil.which(override):
            return Path(found)
        raise BuildError(f"{BROWSER_ENV} is set to '{override}', but that file does not exist")
    candidates = browser_candidates()
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n  ".join(dict.fromkeys(str(c) for c in candidates)) or "(nothing to try)"
    raise BuildError(
        "No Chrome, Chromium or Edge found. Install one of them, or set the environment "
        f"variable {BROWSER_ENV} to the browser executable.\nLooked for:\n  {tried}\n"
        f"and on PATH for: {', '.join(PATH_BROWSERS)}"
    )


def run_browser(browser: Path, arguments: list[str]) -> str:
    """Run the browser once in headless mode with a throw-away profile.

    Args:
        browser: Path of the browser executable.
        arguments: Task-specific flags followed by the URL to open.

    Returns:
        What the browser wrote to standard output.

    Raises:
        BuildError: If the browser cannot be started, times out or reports failure.
    """
    extra = os.environ.get(BROWSER_FLAGS_ENV, "").split()
    with tempfile.TemporaryDirectory(
        prefix="browser-profile-", dir=BUILD_DIR, ignore_cleanup_errors=True
    ) as profile:
        command = [
            str(browser),
            "--headless=new",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-gpu",
            "--hide-scrollbars",
            *extra,
            *arguments,
        ]
        try:
            result = subprocess.run(
                command, capture_output=True, timeout=BROWSER_TIMEOUT_S, check=False
            )
        except subprocess.TimeoutExpired as error:
            raise BuildError(
                f"{browser.name} did not finish within {BROWSER_TIMEOUT_S} seconds"
            ) from error
        except OSError as error:
            raise BuildError(f"Could not start {browser}: {error}") from error
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()[-2000:]
        raise BuildError(f"{browser.name} exited with code {result.returncode}:\n{details}")
    return result.stdout.decode("utf-8", errors="replace")


def sha256_of(path: Path) -> str:
    """Return the SHA-256 checksum of a file as hexadecimal text."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_mermaid() -> Path:
    """Return the cached copy of the pinned mermaid.js, downloading it once.

    The file is checked against ``MERMAID_SHA256`` on every build, so a
    damaged or replaced file is never run.

    Returns:
        Path of the verified file in ``build/cache/``.

    Raises:
        BuildError: If the download fails or the checksum is wrong.
    """
    target = CACHE_DIR / f"mermaid-{MERMAID_VERSION}.min.js"
    if not target.is_file():
        log.info("Downloading mermaid.js %s (once; later builds work offline)", MERMAID_VERSION)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        try:
            with urllib.request.urlopen(MERMAID_URL, timeout=60) as response:
                partial.write_bytes(response.read())
        except (urllib.error.URLError, TimeoutError) as error:
            raise BuildError(
                f"Could not download {MERMAID_URL}: {error}\n"
                f"Download it by hand and save it as {target}"
            ) from error
        partial.replace(target)
    actual = sha256_of(target)
    if actual != MERMAID_SHA256:
        target.unlink()
        raise BuildError(
            f"Checksum of the cached mermaid.js is wrong (expected {MERMAID_SHA256}, "
            f"got {actual}). The file was deleted; run the build again to fetch it."
        )
    return target


def diagram_page(diagrams: list[Diagram], mermaid_js: Path) -> str:
    """Write the HTML page that renders all diagrams inside the browser."""
    payload = json.dumps([{"index": d.index, "code": d.code} for d in diagrams])
    payload = payload.replace("</", "<\\/")  # must not end the <script> element early
    script = DIAGRAM_SCRIPT.read_text(encoding="utf-8")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        "<title>Book diagrams</title>\n</head>\n<body>\n"
        '<div id="host" style="width: 2000px"></div>\n'
        f"<script>window.BOOK_DIAGRAMS = {payload};</script>\n"
        f'<script src="{mermaid_js.as_uri()}"></script>\n'
        f"<script>\n{script}\n</script>\n</body>\n</html>\n"
    )


def check_browser_version(user_agent: str, browser: Path) -> None:
    """Stop if the browser is too old to print page numbers.

    Args:
        user_agent: User-agent text reported by the browser.
        browser: Path of the browser, for the message.

    Raises:
        BuildError: If the Chromium version is below ``MIN_BROWSER_MAJOR``.
    """
    match = re.search(r"Chrome/(\d+)", user_agent)
    if match and int(match.group(1)) < MIN_BROWSER_MAJOR:
        raise BuildError(
            f"{browser} is Chromium {match.group(1)}. Version {MIN_BROWSER_MAJOR} or newer "
            "is needed for page numbers and running headers. Update the browser, or set "
            f"{BROWSER_ENV} to a newer one."
        )


def render_diagrams(book: Book, browser: Path) -> dict[int, dict[str, object]]:
    """Render every Mermaid diagram to SVG in the browser.

    Args:
        book: The book with its collected diagrams.
        browser: Path of the browser executable.

    Returns:
        One report entry per diagram index, with the keys ``ok``, ``svg``,
        ``error``, ``width`` and ``height``.

    Raises:
        BuildError: If rendering did not finish or mermaid.js could not run.
    """
    mermaid_js = ensure_mermaid()
    page = BUILD_DIR / "diagrams.html"
    page.write_text(diagram_page(book.diagrams, mermaid_js), encoding="utf-8")
    dom = run_browser(
        browser,
        [f"--virtual-time-budget={DIAGRAM_TIME_BUDGET_MS}", "--dump-dom", page.as_uri()],
    )
    match = REPORT_RE.search(dom)
    if match is None:
        raise BuildError(
            "The browser did not finish rendering the Mermaid diagrams (no report in the "
            f"page). Open {page} in a browser and check its console."
        )
    report = json.loads(html.unescape(match.group(1)))
    check_browser_version(str(report.get("userAgent", "")), browser)
    if report.get("fatal"):
        raise BuildError(f"Mermaid could not run: {report['fatal']}")
    results = {int(entry["index"]): entry for entry in report["diagrams"]}
    missing = [d.location for d in book.diagrams if d.index not in results]
    if missing:
        raise BuildError("No render result for: " + "; ".join(missing))
    return results


def figure_html(diagram: Diagram, result: dict[str, object], book: Book) -> str:
    """Return the HTML that replaces one diagram placeholder.

    A rendered diagram is scaled down, never up, so that it fits the text
    block and at most about half a page. A failed diagram is shown as its
    source with a note.
    """
    if not result.get("ok"):
        return (
            '<div class="diagram-failed">\n'
            '<p class="diagram-failed-title">Diagram failed to render: '
            f"{html.escape(str(result.get('error')))}</p>\n"
            f'<pre class="code"><code>{html.escape(diagram.code)}</code></pre>\n</div>'
        )
    width = float(str(result["width"]))
    height = float(str(result["height"]))
    scale = min(DIAGRAM_BASE_SCALE, TEXT_WIDTH_PX / width, MAX_DIAGRAM_HEIGHT_PX / height)
    label_pt = DIAGRAM_FONT_PT * scale
    if label_pt < MIN_LABEL_PT:
        book.warn(
            f"{diagram.location} is too large for the page and is shrunk until its labels "
            f"are about {label_pt:.1f} pt; shorter labels, more line breaks in the labels "
            "or fewer boxes in one row would make it readable in print"
        )
    return f'<div class="diagram-box" style="width: {width * scale:.1f}px">{result["svg"]}</div>'


def insert_diagrams(book: Book, browser: Path, *, allow_broken: bool) -> None:
    """Render the diagrams and put them into the section HTML.

    Args:
        book: The book; the ``body`` of its sections is updated.
        browser: Path of the browser executable.
        allow_broken: If true, a diagram that fails is printed as source code
            with a visible note instead of stopping the build.

    Raises:
        BuildError: If a diagram fails and ``allow_broken`` is false.
    """
    results = render_diagrams(book, browser)
    failures = [d for d in book.diagrams if not results[d.index].get("ok")]
    if failures:
        lines = [f"  {d.location}: {results[d.index].get('error')}" for d in failures]
        message = f"{len(failures)} Mermaid diagram(s) failed to render:\n" + "\n".join(lines)
        if not allow_broken:
            raise BuildError(
                message + "\nFix the Markdown, or run again with --allow-broken-diagrams "
                "to get a review PDF that shows the diagram source instead."
            )
        book.warn(message)
    by_source: dict[Source, list[Diagram]] = {}
    for diagram in book.diagrams:
        by_source.setdefault(diagram.source, []).append(diagram)
    for section in book.sections:
        for diagram in by_source.get(section.source, []):
            placeholder = DIAGRAM_PLACEHOLDER.format(index=diagram.index)
            figure = figure_html(diagram, results[diagram.index], book)
            section.body = section.body.replace(placeholder, figure)
    log.info("Rendered %d of %d diagrams", len(book.diagrams) - len(failures), len(book.diagrams))


# ---------------------------------------------------------------------------
# Step 4 - one HTML document, printed to PDF
# ---------------------------------------------------------------------------


def git_revision(output: Path) -> str:
    """Describe the commit the book is built from.

    Args:
        output: The PDF being written; it is ignored when checking for changes,
            because the build itself modifies it.

    Returns:
        For example ``"41e5ef1"`` or ``"41e5ef1 + uncommitted changes"``, or
        ``"unknown"`` if git is not available.
    """

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "--short", "HEAD")
        pathspec = ["--", "."]
        if output.resolve().is_relative_to(REPO_ROOT):
            relative = output.resolve().relative_to(REPO_ROOT).as_posix()
            pathspec.append(f":(exclude){relative}")
        dirty = git("status", "--porcelain", *pathspec)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit} + uncommitted changes" if dirty else commit


def css_string(text: str) -> str:
    """Quote text for use as a CSS string, for example in ``content:``."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
    return f'"{escaped}"'


def split_book_title(cover_title: str) -> tuple[str, str]:
    """Split the cover heading at its first colon into title and subtitle."""
    title, _, subtitle = cover_title.partition(":")
    subtitle = subtitle.strip()
    return title.strip(), subtitle[:1].upper() + subtitle[1:]


def page_css(book: Book, book_title: str) -> str:
    """Generate the ``@page`` rules: paper size, margins, headers and page numbers.

    Each section is printed on its own named page so that its title can be
    shown in the header. The title page has no header and no page number.

    Args:
        book: The book, for the section titles.
        book_title: Short title shown at the top left of every page.

    Returns:
        CSS text.
    """
    margin_box = (
        'font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; '
        "font-size: 8.5pt; color: #5b6672;"
    )
    rules = [
        "@page {",
        "  size: A4;",
        f"  margin: {MARGIN_TOP_MM}mm {MARGIN_SIDE_MM}mm {MARGIN_BOTTOM_MM}mm;",
        f"  @top-left {{ content: {css_string(book_title)}; {margin_box} }}",
        f"  @top-right {{ content: ''; {margin_box} }}",
        f"  @bottom-center {{ content: counter(page); {margin_box} font-size: 9.5pt; }}",
        "}",
        "@page cover {",
        "  margin: 0;",
        "  @top-left { content: none; }",
        "  @top-right { content: none; }",
        "  @bottom-center { content: none; }",
        "}",
        f"@page contents {{ @top-right {{ content: {css_string('Contents')}; }} }}",
    ]
    for number, section in enumerate(book.sections):
        title = COVER_TITLE if section.source.kind == "cover" else section.title
        rules.append(f"@page section{number} {{ @top-right {{ content: {css_string(title)}; }} }}")
    return "\n".join(rules) + "\n"


def cover_html(title: str, subtitle: str, revision: str, today: dt.date) -> str:
    """Return the HTML of the title page."""
    built = f"{today.day} {today:%B %Y}"
    return (
        '<div class="cover" style="page: cover">\n'
        f'<p class="cover-kicker">{html.escape(REPOSITORY_NAME)}</p>\n'
        f"<h1>{html.escape(title)}</h1>\n"
        f'<p class="cover-subtitle">{html.escape(subtitle)}</p>\n'
        '<p class="cover-build">'
        f"Built on <strong>{built}</strong><br>"
        f"From commit <strong>{html.escape(revision)}</strong></p>\n"
        "</div>\n"
    )


def toc_groups(book: Book) -> list[tuple[str, list[Section]]]:
    """Group the chapters and appendices into the parts named in the cover file.

    A file that the cover file does not list yet joins the part of the file
    before it, so a brand-new chapter still appears in the contents.

    Args:
        book: The book with ``parts`` filled in.

    Returns:
        Part names with their sections, in reading order.
    """
    defaults = {"chapter": "Chapters", "appendix": "Appendices"}
    groups: list[tuple[str, list[Section]]] = []
    previous: Section | None = None
    for section in book.sections[1:]:
        label = book.parts.get(section.source.path)
        if label is None:
            book.warn(f"{section.source.path.name} is not listed in {COVER_FILE}")
            same_kind = previous is not None and previous.source.kind == section.source.kind
            label = groups[-1][0] if same_kind else defaults[section.source.kind]
        if groups and groups[-1][0] == label:
            groups[-1][1].append(section)
        else:
            groups.append((label, [section]))
        previous = section
    return groups


def toc_entry(anchor: str, title: str, pages: dict[str, int] | None) -> str:
    """Return one line of the table of contents: title, dots, page number."""
    page = TOC_PAGE_PLACEHOLDER if pages is None else str(pages.get(anchor, ""))
    return (
        f'<a class="toc-entry" href="#{anchor}"><span class="toc-title">{html.escape(title)}'
        f'</span><span class="toc-dots"></span><span class="toc-page">{page}</span></a>\n'
    )


def toc_html(book: Book, pages: dict[str, int] | None) -> str:
    """Return the table of contents.

    Args:
        book: The book.
        pages: Page number of each section anchor, or ``None`` on the first
            print pass, when the numbers are not known yet.

    Returns:
        The contents section as HTML.
    """
    parts = [
        '<section id="contents" class="toc" style="page: contents">\n',
        '<h1 class="section-title">Contents</h1>\n',
        toc_entry(book.sections[0].source.anchor, COVER_TITLE, pages),
    ]
    for label, sections in book.groups:
        parts.append(f'<p class="toc-part">{html.escape(label)}</p>\n')
        parts += [toc_entry(s.source.anchor, s.title, pages) for s in sections]
    parts.append("</section>\n")
    return "".join(parts)


def section_heading(title: str) -> str:
    """Return the opening heading of a section.

    "Chapter 1 - What we are building" is shown as a small "Chapter 1" above a
    large title. The dash stays in the text (hidden by the stylesheet) so that
    the PDF bookmark reads like the original heading.
    """
    match = TITLE_SPLIT_RE.fullmatch(title)
    if match is None:
        return f'<h1 class="section-title">{html.escape(title)}</h1>\n'
    kicker, dash, rest = (html.escape(group) for group in match.groups())
    return (
        f'<h1 class="section-title"><span class="kicker">{kicker}</span>'
        f'<span class="dash">{dash}</span>{rest}</h1>\n'
    )


def section_html(number: int, section: Section) -> str:
    """Return one chapter, appendix or front-matter part as a ``<section>``."""
    title = COVER_TITLE if section.source.kind == "cover" else section.title
    return (
        f'<section id="{section.source.anchor}" class="section-{section.source.kind}" '
        f'style="page: section{number}">\n{section_heading(title)}{section.body}</section>\n'
    )


def book_html(book: Book, revision: str, today: dt.date, pages: dict[str, int] | None) -> str:
    """Assemble the complete HTML document of the book.

    Args:
        book: The book with rendered sections and diagrams.
        revision: Text describing the git commit.
        today: The build date.
        pages: Page numbers for the table of contents, if already known.

    Returns:
        The HTML document.
    """
    title, subtitle = split_book_title(book.sections[0].title)
    pygments_css = HtmlFormatter(style="friendly").get_style_defs(".highlight")
    head = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{html.escape(book.sections[0].title)}</title>\n"
        f"<style>\n{pygments_css}\n</style>\n"
        f"<style>\n{CSS_FILE.read_text(encoding='utf-8')}\n</style>\n"
        f"<style>\n{page_css(book, title)}</style>\n</head>\n<body>\n"
    )
    sections = "".join(section_html(number, s) for number, s in enumerate(book.sections))
    return (
        head
        + cover_html(title, subtitle, revision, today)
        + toc_html(book, pages)
        + sections
        + "</body>\n</html>\n"
    )


def print_pdf(browser: Path, page: Path, pdf: Path) -> None:
    """Print an HTML file to PDF with the browser.

    Args:
        browser: Path of the browser executable.
        page: HTML file to print.
        pdf: PDF file to write.

    Raises:
        BuildError: If the browser did not produce a PDF.
    """
    pdf.unlink(missing_ok=True)
    run_browser(
        browser,
        [
            f"--print-to-pdf={pdf}",
            "--no-pdf-header-footer",
            "--generate-pdf-document-outline",
            page.as_uri(),
        ],
    )
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF"):
        raise BuildError(f"{browser.name} did not write a PDF to {pdf}")


def read_layout(pdf: Path, anchors: list[str]) -> PdfLayout | None:
    """Read the page count and the start page of each section from a PDF.

    Args:
        pdf: The printed PDF.
        anchors: Section anchors to look up.

    Returns:
        The layout, or ``None`` if the optional package ``pypdf`` is not installed.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    reader = PdfReader(pdf)
    # Chrome stores the id "x" as the PDF name "/x".
    destinations = {name.lstrip("/"): dest for name, dest in reader.named_destinations.items()}
    pages = {
        anchor: reader.get_destination_page_number(destinations[anchor]) + 1
        for anchor in anchors
        if anchor in destinations
    }
    return PdfLayout(len(reader.pages), pages, len(reader.outline))


def print_book(book: Book, browser: Path, revision: str, today: dt.date) -> tuple[Path, PdfLayout]:
    """Print the book, then print it again with page numbers in the contents.

    The browser cannot write "see page N" by itself. So the book is printed
    once, the start page of every section is read from that PDF, and the book
    is printed a second time with those numbers. The second PDF is checked:
    the numbers must still be true.

    Args:
        book: The finished book.
        browser: Path of the browser executable.
        revision: Text describing the git commit.
        today: The build date.

    Returns:
        The path of the PDF in the build folder and its layout.

    Raises:
        BuildError: If a section has no page number or the two passes disagree.
    """
    page = BUILD_DIR / "book.html"
    pdf = BUILD_DIR / "book.pdf"
    anchors = [section.source.anchor for section in book.sections]

    page.write_text(book_html(book, revision, today, None), encoding="utf-8")
    print_pdf(browser, page, pdf)
    first = read_layout(pdf, anchors)
    if first is None:
        raise BuildError(
            "The package 'pypdf' is needed for the page numbers in the table of contents. "
            f'Install it with: "{sys.executable}" -m pip install pypdf'
        )
    missing = [anchor for anchor in anchors if anchor not in first.pages]
    if missing:
        raise BuildError(f"The PDF has no link target for: {', '.join(missing)}")

    page.write_text(book_html(book, revision, today, first.pages), encoding="utf-8")
    print_pdf(browser, page, pdf)
    second = read_layout(pdf, anchors)
    if second is None or second.pages != first.pages:
        raise BuildError(
            "The page numbers changed between the two print passes, so the table of "
            "contents would be wrong. Please report this; the HTML is in " + str(page)
        )
    return pdf, second


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(
        description="Build the study book in docs/learning/ into one PDF textbook."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PDF,
        help="PDF file to write (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-broken-diagrams",
        action="store_true",
        help="do not stop when a Mermaid diagram fails to render; print its source with "
        "a visible note instead (for a review copy only)",
    )
    return parser.parse_args(argv)


def build(output: Path, *, allow_broken: bool) -> None:
    """Build the PDF.

    Args:
        output: PDF file to write.
        allow_broken: See ``--allow-broken-diagrams``.

    Raises:
        BuildError: If any step fails.
    """
    started = time.perf_counter()
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    browser = find_browser()
    log.info("Browser: %s", browser)

    sources = discover_sources(BOOK_DIR)
    book = load_book(sources)
    log.info("Read %d Markdown files, %d diagrams", len(sources), len(book.diagrams))

    insert_diagrams(book, browser, allow_broken=allow_broken)
    pdf, layout = print_book(book, browser, git_revision(output), dt.datetime.now().date())

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, output)
    except PermissionError as error:
        raise BuildError(
            f"Cannot write {output}. If the PDF is open in a viewer, close it and run again."
        ) from error

    log.info(
        "Wrote %s: %d pages, %d bookmarks, %.2f MB, %d warning(s), %.1f s",
        output,
        layout.page_count,
        layout.outline_entries,
        output.stat().st_size / 1_000_000,
        len(book.warnings),
        time.perf_counter() - started,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the build from the command line.

    Args:
        argv: Command-line arguments; ``None`` means ``sys.argv``.

    Returns:
        0 on success, 1 if the build failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    arguments = parse_arguments(argv)
    try:
        build(arguments.output.resolve(), allow_broken=arguments.allow_broken_diagrams)
    except BuildError as error:
        log.error("BUILD FAILED: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
