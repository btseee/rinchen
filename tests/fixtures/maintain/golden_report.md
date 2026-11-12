# Maintenance check: drift (nightly cadence)

## Checks

| step | check | state | reason |
| --- | --- | --- | --- |
| doctor | D-PY | ok |  |
| doctor | D-WHEELS | ok |  |
| doctor | D-EOLE-PY | ok |  |
| doctor | D-EOLE-CLI | ok |  |
| doctor | D-CT2-CONV | ok |  |
| doctor | D-CT2-IMPORT | ok |  |
| doctor | D-TORCH-CUDA | skipped | a GPU check: maintain check never asks a device question |
| lock_file | - | ok |  |
| lock_consistent | - | ok |  |
| environment | - | ok |  |
| drift | - | drift |  |
| reruns | - | ok |  |

## Dependency drift

State: drift. 7 entries (5 update, 1 add, 1 remove; 3 drift, 4 blocked).

| package | change | locked | available | class | needs |
| --- | --- | --- | --- | --- | --- |
| huggingface-hub | update | 2.4.0 | 2.4.1 | blocked | evaluation_tiny, gate |
| numpy | update | 2.5.4 | 2.5.5 | blocked | numerics, evaluation_tiny |
| nvidia-cudnn-cu13 | update | 9.24.0.43 | 9.25.0.10 | blocked | parity_tiny, robustness_tiny, gate |
| rich | update | 15.0.0 | 15.0.1 | drift | - |
| safetensors | update | 0.8.0 | 0.8.1 | blocked | gate |
| sortedcontainers | remove | 2.4.0 | - | drift | - |
| tomli-w | add | - | 1.2.0 | drift | - |

The drift table is what uv could take within the constraints of pyproject.toml; a package pinned with == never appears in it (https://github.com/btseee/rinchen/wiki/Runbook-14-Maintenance).

## Parity reruns

| id | state | passed | failed | skipped | reason |
| --- | --- | --- | --- | --- | --- |
| numerics | ok | 1110 | 0 | 0 |  |
| parity_tiny | ok | 13 | 0 | 0 |  |
| parity_export | skipped | - | - | - | no export was given |
| robustness_tiny | skipped | - | - | - | not due at the nightly cadence |
| evaluation_tiny | skipped | - | - | - | not due at the nightly cadence |
| serving_tiny | skipped | - | - | - | not due at the nightly cadence |
