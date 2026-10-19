# Flashcards with Anki

The course ships [Anki](https://apps.ankiweb.net/) flashcards for spaced-repetition
review of each lesson's vocabulary, notation, and short calculations. The decks are
plain-text files in [`flashcards/`](../flashcards/RL-lesson-01-rl-foundations-01-anki.txt)
that Anki imports directly, so they can be read, edited, and reviewed in Git like any
other course file.

| Lesson | File |
| --- | --- |
| Lesson 1 – RL foundations | [`RL-lesson-01-rl-foundations-01-anki.txt`](../flashcards/RL-lesson-01-rl-foundations-01-anki.txt) |

## Import a deck

1. Install Anki (desktop, version 2.1.55 or later, which reads the file headers below).
2. Download the deck file. On GitHub, open the file and use **Download raw file**;
   on the website, follow the link above and save the page.
3. In Anki, choose **File → Import** and select the `.txt` file.
4. Check the import screen. The file's headers already set everything, so the
   defaults should read:
    - **Field separator:** Tab
    - **Allow HTML in fields:** on
    - **Notetype:** Basic
    - **Deck:** `Reinforcement Learning::Lesson 01 - RL Foundations`
    - **Field mapping:** Front, Back, Tags
5. Click **Import**. The deck appears under *Reinforcement Learning* in the deck list.

Formulas are written in Anki's MathJax syntax (`\( … \)` inline, `\[ … \]` displayed)
and render on both desktop and AnkiMobile/AnkiDroid without add-ons.

## Update a deck after the file changes

Anki matches imported notes by their first field (the question). A card whose question
text changed imports as a **new** note beside the old one. To pick up a reworked deck
cleanly, delete the old lesson deck first (**right-click the deck → Delete**), then
import the new file. Deleting a deck also deletes its review history.

If only the answers changed, importing over the existing deck with **Existing notes →
Update** keeps your review history.

## Study tips

- Review daily; Anki schedules each card from how well you recalled it.
- Answer out loud or on paper before flipping. For calculation cards, write the
  arithmetic.
- Use tags to focus a session: in the browser, search `tag:practice` for calculation
  cards or `tag:notation` for symbol reading.

## Write new cards

Each deck file starts with Anki header lines, followed by one card per line:

```text
#separator:Tab
#html:true
#notetype:Basic
#deck:Reinforcement Learning::Lesson 01 - RL Foundations
#tags:RL lesson01 foundations
#columns:Front	Back	Tags
#tags column:3
What is a reward?	A number scoring one transition. It does not say which action would have been best.	reward
```

The three columns are separated by a **tab character**, not spaces:

- **Front**: one question asking for one fact.
- **Back**: the answer in one or two sentences. HTML such as `<br>` and `<b>` is allowed;
  write a literal `<` as `&lt;`.
- **Tags**: space-separated topic tags, matching the [glossary](glossary.md) terms
  where possible. Add `practice` for calculations and `notation` for symbol reading.
  The `#tags:` header adds the lesson tags to every card.

For a new lesson, copy the header block, change the `#deck:` and `#tags:` lines, and
save the file as `flashcards/RL-lesson-NN-<topic>-01-anki.txt`. Keep each card to a
single tab-separated line; this one-liner reports any line with the wrong number of
fields:

```bash
awk -F'\t' '!/^#/ && NF != 3 {print FILENAME ":" FNR}' flashcards/*.txt
```

Good cards follow the lesson closely: test one idea, avoid examples that need the
lesson open to make sense, and prefer a small worked calculation over a definition
when the lesson includes one.
