# Accessibility

We want everyone, including people with disabilities and people who use assistive technology, to be able to read, use and contribute to Rinchen. Rinchen is a machine translation pipeline with Mongolian at its center, so the people who read it include those who read Mongolian in Cyrillic or in the traditional script, and those who read English as a second language.

This statement says what it covers, what the project does, what is known to fall short, how to report a barrier, and what to expect. It was written on 2026-11-12 and describes the repository as it was on that day.

## Our commitment

We treat accessibility as part of quality: a document, a message or a report that someone cannot read is not finished. We treat a report of a barrier as expertise, not as a complaint.

This is a commitment to work on it, not a claim that the work is done. The limits we know about are listed below.

## Scope

This statement covers the text the project writes for people:

- the documentation: the README, the guides in the repository and the [project Wiki](https://github.com/btseee/rinchen/wiki)
- the output of the commands in a terminal, including the `--help` text
- the reports the commands write, such as `report.md` of the tiny pipeline, of an evaluation and of the maintenance check
- the issue forms and the pull request template
- the error messages of the HTTP service that `rinchen serve` starts

It does not cover GitHub's own pages and tools, such as how GitHub shows the Wiki or an issue form, the websites we link to, or the output of the tools the project calls (uv, pytest, Eole and others), which follows their own formats.

## What the project does

In the documents:

- The community files are written in plain ASCII text, with short paragraphs, headings in order and links whose text says where they go. A test checks this: `tests/unit/test_github_community_files.py`.
- Lists are real lists, and tables have header rows.
- The one image in the repository is the logo, `images/logo.png`, and the README gives it a text alternative.

In the output:

- The commands write plain lines of text. None sets a color, draws a progress bar or draws boxes of its own, and the code of the commands holds no character outside ASCII.
- A state is written as a word, never by color alone: `rinchen doctor` marks each check `[OK]`, `[FAIL]` or `[SKIP]`, and the reports write it as a word such as `ok`, `pass`, `fail` or `skipped`.
- The reports are Markdown with header rows in their tables. The text that goes into the tiny pipeline's report is checked to be printable ASCII.
- The HTTP service answers every error in the same JSON shape: a short code and a message in plain English that names the field or the limit.

## Known limitations

- The project has not been evaluated by an accessibility audit, and nobody has tested its documents or output with a screen reader, a magnifier or speech input. What is described above comes from how the files are written, not from tests with assistive technology.
- Rinchen can write traditional Mongolian script (the language code is `mn_Mong`). That script is written vertically, from top to bottom, and needs a font and a text renderer that support it. In a terminal, a Markdown file or an issue it is shown horizontally and may appear as separate or wrongly joined letters.
- The project has not checked how its traditional Mongolian script output looks in any font, and no Mongolian reader has judged the converter's output. We do not know whether screen readers read this script correctly.
- Mongolian words in Cyrillic appear in the README and elsewhere with no language marked, because Markdown cannot mark the language of a phrase. A screen reader may read them in the voice of the page's language.
- The `--help` pages and the usage errors of the commands are drawn by the library that formats them, with box-drawing characters, and may use color in a terminal that supports it. A screen reader may read the box characters aloud.
- The README ends with one "back to top" link that is written as raw HTML. A screen reader may announce it differently from the other links.
- Some reports have wide tables: the evaluation report has tables of up to ten columns. A screen reader reads them cell by cell, and a narrow screen needs horizontal scrolling.
- The documentation and the messages are in English only. None of it has been translated into Mongolian.
- Some Wiki pages (the runbooks, the Verification Log and the package list) still carry an interim notice: they are being rewritten to describe the current state only, and until then some sentences may be out of date.

## Standards

We aim for the Web Content Accessibility Guidelines (WCAG) 2.2, Level AA, for the documents we write. Read [WCAG 2.2](https://www.w3.org/TR/WCAG22/) for the full text.

This is a goal. The project has not been evaluated, so we do not claim conformance for any document. If an evaluation is made, its scope, date, method and evaluator will be recorded here.

## Writing with accessibility in mind

If you write documentation, messages or output for this project:

- Use plain language and short sentences.
- Put headings in order and do not skip a level. Use real lists, and give tables header rows.
- Write link text that says where the link goes. Do not write "here" or "click here".
- Give every image a text alternative. For a diagram, describe it in text nearby.
- Do not carry meaning by color alone. Write the state as a word.
- Keep the output of commands to plain ASCII lines that read well from top to bottom.

The pull request template asks whether you considered the accessibility of new text or output.

## Reporting a barrier

If you meet a barrier, please tell us. There are two ways:

1. Open an issue with the [accessibility issue form](https://github.com/btseee/rinchen/issues/new?template=accessibility.yml).
2. Write to bbattseren88@gmail.com.

When you can, say what you were trying to do and what went wrong, where it happened (a page, a file or a command), and the steps to reproduce it. Add your operating system, your terminal or browser, and your assistive technology, with versions. A screenshot or a recording is welcome but not needed.

You do not have to say why you use an assistive tool, and you do not have to share any medical or personal detail.

### Severity

The form lets you suggest a severity, and the maintainer confirms it:

- Critical: you cannot complete a core task.
- Serious: it is very hard, but there is a workaround.
- Moderate: it is annoying or inconsistent.
- Minor: it has little effect.

## What to expect

- We aim to acknowledge your report within 7 days and to give a first assessment within 14 days: how serious we judge it, what happens next, and a workaround if there is one.
- We will explain next steps, known limits and anything that depends on another project.
- We will update the issue when something changes, and we may ask you to confirm that a fix works for you.
- The project has one maintainer, so we set no fixed resolution time for each severity. We say in the issue when we expect to fix it, and tell you if that changes.

## Ownership and feedback

Accessibility is owned by the project maintainer (the GitHub account `btseee`). The maintainer triages reports, keeps the list of known limitations current, and reviews this statement when something it describes changes.

To suggest an improvement to this statement or to our practices, open an issue or a pull request. For a barrier you are meeting now, use the steps in "Reporting a barrier".
