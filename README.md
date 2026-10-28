<a id="readme-top"></a>

<div align="center">
  <a href="https://github.com/btseee/rinchen">
    <img src="./images/logo.png" alt="Rinchen logo" width="120" height="120">
  </a>

  <h3 align="center">Rinchen</h3>

  <p align="center">
    Open-source, Mongolian-centric machine translation trained from scratch.
    <br />
    Context-aware translations, fast inference, and CPU-friendly deployment.
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#features">Features</a></li>
    <li><a href="#supported-languages">Supported Languages</a></li>
    <li><a href="#models">Models</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#evaluation">Evaluation</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#development">Development</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

## About

Rinchen is an open-source machine translation system with Mongolian (Khalkha, Cyrillic) as its central language. It is trained from random initialization for 16 translation directions between Mongolian and English, Russian, Chinese, Korean, Japanese, German, Kazakh, and Turkish.

The system is designed for context-aware translation, preservation of numbers, dates, names, and terminology, and efficient inference on both CPU and GPU.

About Byamba Rinchen was a renowned Mongolian writer, linguist, translator, and scholar who dedicated his life to preserving Mongolia’s language, history, and cultural heritage. He is regarded as a pioneer of modern Mongolian literature, best known for his historical novel Үүрийн туяа (Rays of Dawn) and his contributions to Mongolian linguistics. [Wikipedia](https://de.wikipedia.org/wiki/Bjambyn_Rintschen)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Features

- Mongolian-centric multilingual machine translation
- 16 translation directions
- Mongolian Cyrillic and traditional Mongolian script support
- Document-level context handling
- Placeholder and terminology preservation
- Teacher, server, and edge model variants
- CTranslate2 INT8 artifacts for fast CPU inference
- REST API, CLI, and Python package
- Evaluation against NLLB-200 3.3B using chrF++, BLEU, COMET-22, and MetricX-24

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Supported Languages

| Language | Code | Target tag |
|---|---|---|
| Mongolian, Cyrillic | `mn` | `<2mn>` |
| Mongolian, traditional script | `mn_Mong` | `<2mn_Mong>` |
| English | `en` | `<2en>` |
| Russian | `ru` | `<2ru>` |
| Chinese | `zh` | `<2zh>` |
| Korean | `ko` | `<2ko>` |
| Japanese | `ja` | `<2ja>` |
| German | `de` | `<2de>` |
| Kazakh | `kk` | `<2kk>` |
| Turkish | `tr` | `<2tr>` |

Supported directions:

```text
mn  <-> en, ru, zh, ko, ja, de, kk, tr
```

Traditional Mongolian script is supported as an output target through the `<2mn_Mong>` tag.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Models

| Model | Parameters | Purpose |
|---|---:|---|
| Teacher | ~1.3B | Highest-quality translation model |
| Server student | ~190M | Balanced quality and serving performance |
| Edge student | ~70M | CPU and resource-constrained deployment |

All models are trained from random initialization. External pretrained models are used only for evaluation baselines or synthetic-data generation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- C++ build tools, required for optional Korean and Japanese tokenizer components
- GPU with CUDA support for training
- CPU-only environment for inference with the edge model

### Installation

```bash
git clone [https://github.com/btseee/rinchen.git](https://github.com/btseee/rinchen.git)
cd rinchen
uv sync
```

Verify the environment:

```bash
uv run python -c "import sys; assert sys.version_info[:2] == (3, 14)"
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### CLI

Translate text from Mongolian to English:

```bash
uv run rinchen translate --src mn --tgt en < input.txt
```

Translate from English to Mongolian:

```bash
uv run rinchen translate --src en --tgt mn < input.txt
```

Translate into traditional Mongolian script:

```bash
uv run rinchen translate --src en --tgt mn_Mong < input.txt
```

### Python

```python
from rinchen import Translator

translator = Translator(
    model_path="models/server-ct2",
    source_language="mn",
    target_language="en",
)

result = translator.translate("Сайн байна уу?")
print(result)
```

### API

Start the local API server:

```bash
uv run uvicorn rinchen.serve.app:app --host 0.0.0.0 --port 8000
```

Translate through the API:

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "source_language": "mn",
    "target_language": "en",
    "text": "Сайн байна уу?"
  }'
```

Available endpoints:

```text
POST /translate
GET  /languages
GET  /health
POST /feedback
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Evaluation

Rinchen is evaluated on FLORES+ devtest and Mongolia-sourced human evaluation sets.

Primary metrics:

- chrF++
- BLEU
- COMET-22
- MetricX-24
- Off-target language rate
- Number and date preservation
- Placeholder and terminology compliance
- Translation latency and throughput

Release targets:

- Beat NLLB-200 3.3B on chrF++ in at least 12 of 16 directions on FLORES+ devtest
- Off-target rate at most 1%
- Number and date preservation at least 99%
- Placeholder and tag validity of 100%

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

```text
.
├── configs/
│   ├── spm/
│   ├── eole/
│   └── serve/
├── src/
│   └── rinchen/
│       ├── align/
│       ├── cli/
│       ├── convert/
│       ├── data/
│       ├── eval/
│       ├── filter/
│       ├── metrics/
│       ├── serve/
│       ├── synth/
│       └── text/
├── tests/
├── scripts/
├── plans/
├── reports/
├── manifests/
├── annotation_app/
├── data/
└── .github/
    └── workflows/
        └── ci.yml
```

The `data/` directory is intended to remain local and is excluded from version control.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Development

Run tests:

```bash
uv run pytest -q
```

Run linting:

```bash
uv run ruff check
```

Run type checking:

```bash
uv run mypy src
```

Development workflow:

1. Read `STATE.md` before starting work.
2. Create a plan in `plans/phase_N.md`.
3. Write tests before implementation.
4. Implement the change.
5. Run tests, linting, and type checks.
6. Record decisions in `DECISIONS.md`.
7. Add package versions and checkpoint hashes to `manifests/`.
8. Open a focused pull request.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Define 16-direction Mongolian-centric translation scope
- [x] Establish benchmark and baseline evaluation plan
- [ ] Complete data acquisition and inventory
- [ ] Build Mongolia-sourced human test sets
- [ ] Train normalization and tokenizer pipeline
- [ ] Train Cyrillic-to-traditional-script converter
- [ ] Generate and filter synthetic training data
- [ ] Train teacher model
- [ ] Add document-level training
- [ ] Distill server and edge student models
- [ ] Release CTranslate2 INT8 artifacts
- [ ] Publish API, CLI, and Python package
- [ ] Establish feedback-driven maintenance loop

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/short-description
   ```

3. Make focused changes.
4. Add or update tests.
5. Run:

   ```bash
   uv run pytest -q
   uv run ruff check
   uv run mypy src
   ```

6. Commit using a conventional message:

   ```bash
   git commit -m "feat(cli): add traditional-script translation flag"
   ```

7. Push the branch and open a pull request.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Specify the project license here.

<p align="right">(<a href="#readme-top">back to top</a>)</p>