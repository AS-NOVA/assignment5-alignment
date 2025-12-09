# CS336 Spring 2025 Assignment 5: Alignment

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
``` sh
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.

To connect your implementation to the tests, complete the functions in [./tests/adapters.py](./tests/adapters.py).



