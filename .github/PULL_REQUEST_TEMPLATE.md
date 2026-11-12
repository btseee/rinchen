## Summary

<!-- What does this change do, and why? Name the issue it closes, if there is one. Keep it short and plain. -->

## Type of change

<!-- Mark every box that applies by changing [ ] to [x]. -->

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation only
- [ ] Refactor with no change in behavior
- [ ] Tests, scripts, CI or other tooling
- [ ] Accessibility improvement

## What was verified

<!-- List the commands you ran and what they printed. For a fix, show the failing run of the new test first, then the passing run. -->

- `just gate`: <!-- the result line, for example "25243 passed, 1 skipped, 10 deselected" -->
- Other commands:

## Checklist

- [ ] The tests were written first, and a failing test is shown for a fix.
- [ ] No numeric threshold was lowered, and no test was weakened, skipped or deleted.
- [ ] The fixtures the tests read are committed files that `.gitignore` does not hide.
- [ ] The documentation and the Wiki are updated where behavior changed, or the page to change is named here.
- [ ] The project rules in the contributing guide are respected.
- [ ] The accessibility of any new user-facing text or output was considered: plain language, headings in order, descriptive link text, and no information carried by color alone.
