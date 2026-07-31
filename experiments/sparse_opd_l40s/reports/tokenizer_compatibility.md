# Tokenizer compatibility

- Status: **PASS**
- Student: `/opt/sparse-opd/models/Qwen3-4B`
- Teacher: `/opt/sparse-opd/models/Qwen3-Coder-Next-FP8`
- Student vocab size: 151643
- Teacher vocab size: 151643
- Deterministic token IDs checked: 500
- Token-ID mismatches: 0
- Added vocab equal: True

## Probes

| Probe | Result |
|---|---|
| `Hello world` | PASS |
| `L'intelligenza artificiale` | PASS |
| `def fibonacci(n: int) -> int:\n    return n` | PASS |
| `é à è ò ù` | PASS |
| `JSON: {"value": 42}` | PASS |
| `tabs,	spaces, newline\nand triple backticks\n```python\nprint('ok')\n```` | PASS |
| `Unicode: Ελληνικά, русский, 中文, العربية, 👩🏽‍💻` | PASS |

## Failures

None.
