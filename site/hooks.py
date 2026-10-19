"""MkDocs hooks for the course site."""

import re

REPO_BLOB = "https://github.com/GianmarcoRasi/learning-RL/blob/main/"

# Links from lesson pages to Python sources, e.g. ](../src/learning_rl/lessons/lesson02.py)
SOURCE_LINK = re.compile(r"\]\((?:\.\./)*(src/[^)\s]+)\)")
# index.md is the README, whose links to itself are written as ../README.md
README_LINK = re.compile(r"\]\((\.\./)README\.md")


def on_page_markdown(markdown, page, **kwargs):
    """Point source-code links at GitHub and README links at the home page."""
    markdown = SOURCE_LINK.sub(lambda m: f"]({REPO_BLOB}{m.group(1)})", markdown)
    return README_LINK.sub(r"](\1index.md", markdown)
