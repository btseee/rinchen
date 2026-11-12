# Security policy

## Supported versions

Rinchen has not been released: its version is 0.0.0 and no version is tagged. Only the `main` branch is supported, and fixes go to `main`.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| Any other branch or tag | No |

## Reporting a vulnerability

Please report a vulnerability in private. Do not open a public issue or pull request for it.

1. Preferred: use GitHub's private vulnerability reporting. [Report a vulnerability privately](https://github.com/btseee/rinchen/security/advisories/new); only you and the maintainer can see the report.
2. Alternative: write to bbattseren88@gmail.com. Do not put tokens, passwords or personal data in the first message.

## What to include

- The part that is affected: a file, a command, an endpoint or a workflow.
- The version: the commit (`git rev-parse HEAD`) and what you ran it on.
- Steps that reproduce the problem, or a small proof of concept.
- What an attacker could do, and what you expected to happen instead.
- A suggested fix, if you have one.

Remove real secrets and personal data from logs before you send them.

## What to expect

- Within 7 days we acknowledge your report.
- Within 14 days you get a first assessment: whether we can reproduce the problem, how serious we judge it, and what happens next.
- We use coordinated disclosure. We ask you to keep the details private until a fix is on `main`, or until we agree a date with you. Then we publish an advisory and credit you if you wish.
- The project is run by one maintainer and has no bounty program.

## Scope

In scope:

- The code in this repository, such as `src/rinchen`, `scripts` and `docker`.
- The HTTP service that `rinchen serve` starts. It has no authentication and it listens on 127.0.0.1 unless told otherwise, as its help text says, so a report that it answers without credentials is a known property and not a vulnerability. A way around its limits on body size, text length or queue length, a way to read or write files, or a way to crash it is in scope.
- The GitHub workflows in `.github/workflows`.
- The release and upload tooling, `rinchen release stage` and `rinchen release hub`, and the way they handle tokens.

Out of scope:

- The quality or accuracy of a model's translations.
- Third-party packages themselves; report those upstream. A way that this project uses one unsafely is in scope.
- Problems that appear only on a GPU. The tests, the pipeline and the service of this project are run on the CPU.

## Safe practices of the project

You can rely on these, and a report that finds one of them untrue is welcome.

- No secrets are kept in the repository, and no workflow uses a repository secret.
- The code never opens a token file and never logs in on its own. The upload reads its token from the environment variable `HF_TOKEN`, hands it to the Hub client and never prints or writes it.
- `rinchen release hub` is a dry run. Nothing is uploaded unless you pass `--upload`, and an upload needs `HF_TOKEN`. The repository it creates is private unless you pass `--public`.
- Before an upload is planned, the staged text files are scanned, and the command refuses any file that looks like a token or holds an absolute local path.
- The child processes of the tiny pipeline and of the maintenance check do not inherit variables whose names end in `_TOKEN`, `_SECRET`, `_PASSWORD` or `_API_KEY`, and they run with the Hugging Face libraries in offline mode.
- The tiny pipeline needs no network, and its only socket is a loopback listener. Tests that need a network carry the `network` marker, and `just gate` leaves them out.
- The release, nightly and tiny jobs of the workflows ask for read-only access to the repository contents and nothing more, and CI installs exactly the locked dependencies with `uv sync --frozen`.
