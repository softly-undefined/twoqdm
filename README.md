# twoqdm

`twoqdm` is a drop-in `tqdm` wrapper for long-running Python loops. It keeps
the normal progress bar and API while adding a live terminal panel with recent
`it/s` history, colored trend direction, jitter, a smarter ETA hint, and ASCII
loading animations.

## Demo

![twoqdm terminal progress demo](https://raw.githubusercontent.com/softly-undefined/twoqdm/main/twoqdm-demo.gif)

## Install

```bash
python3 -m pip install twoqdm
```

For local development from this repository:

```bash
python3 -m pip install -e ".[dev]"
```

## Use

Replace the import and leave the rest of your existing `tqdm` code unchanged:

```python
# Before: from tqdm import tqdm
from twoqdm import tqdm

for record in tqdm(records, desc="processing"):
    process(record)
```

`trange` is available as a matching drop-in shortcut:

```python
from twoqdm import trange

for i in trange(500, desc="training"):
    train_step(i)
```

ASCII animations are enabled by default. Each progress bar independently
chooses one of the built-in animations using system randomness:

| Name | Animation |
| --- | --- |
| `coffee` | Steam curls above a hot cup |
| `train` | A locomotive rolls over moving track |
| `conveyor` | A continuous line of packages travels down a conveyor |
| `sand-pile` | A sand pile grows with the bar's completion percentage |

Pin a specific animation, request another random choice, or disable it:

```python
tqdm(records, ascii_spinner="conveyor")
tqdm(records, ascii_spinner="random")
tqdm(records, ascii_spinner=False)
```

Inspect the available names and adjust looping frame speed in seconds:

```python
from twoqdm import available_ascii_spinners, trange

print(available_ascii_spinners())

for i in trange(
    500,
    desc="training",
    ascii_spinner="conveyor",
    ascii_spinner_interval=0.10,
):
    train_step(i)
```

Unlike the looping animations, `sand-pile` is tied to progress and reaches its
fullest frame when the bar reaches 100%:

```python
for record in tqdm(records, ascii_spinner="sand-pile"):
    process(record)
```

Preview one from this checkout with:

```bash
python3 examples/show_twoqdm.py --ascii-spinner conveyor --quick
```

The responsive panel hides the animation when the terminal is narrower than
90 columns so the ETA details remain readable.

Terminal resizing is automatic for TTY output. On the next progress update,
both the `tqdm` line and trend panel clear and redraw for the new row and column
count. Pass `ncols=N` or `dynamic_ncols=False` to keep a fixed bar width.

Disable color output with:

```bash
TQDM_TREND_NO_COLOR=1 python3 your_script.py
```

## Build

```bash
rm -rf build dist src/*.egg-info
python3 -m pytest
python3 -m build
python3 -m twine check dist/*
```

## Publish

Create a PyPI API token, then upload:

```bash
python3 -m twine upload dist/*
```

PyPI does not allow re-uploading the same version. Bump the version in
`pyproject.toml` and `src/twoqdm/__init__.py` before publishing a new release.
