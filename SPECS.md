# MLLog Specification

## Overview

Minimal Python logging library for machine learning training loops. Wraps tqdm for progress bars, provides colored status messages, and controls the training loop.

---

## Installation

```bash
uv add tqdm
```

Single file: `mllog.py`

---

## Dependencies

- `tqdm`
- Python stdlib only otherwise

---

## Class: `TrainLogger`

### Constructor

```python
TrainLogger(
    name: str = "Training",
    epochs: int = 1,
    bar_width: int = 40,
)
```

| Param       | Description                   |
| ----------- | ----------------------------- |
| `name`      | Display name for training run |
| `epochs`    | Total epoch count             |
| `bar_width` | Progress bar character width  |

---

## Class: `Colors`

```python
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
```

---

## Methods

### Loop Control

#### `epochs() -> Iterator[int]`

Yields epoch index. Prints epoch header. Controls training loop.

```python
for epoch in logger.epochs():
    ...
```

---

### Progress Bars

#### `train(iterable: Iterable[T]) -> Iterator[T]`

Wraps iterable with progress bar labeled "train".

#### `val(iterable: Iterable[T]) -> Iterator[T]`

Wraps iterable with progress bar labeled "val".

Both preserve the type of items yielded (generic).

---

### Metrics

#### `log(**metrics: float | int | str) -> None`

Prints metrics for current epoch. No color.

---

### Status Messages

All status methods have signature:

```python
def method(message: str, color: Colors = <default>) -> None
```

| Method    | Symbol | Default Color   |
| --------- | ------ | --------------- |
| `info`    | `>`    | `Colors.CYAN`   |
| `warn`    | `!`    | `Colors.YELLOW` |
| `error`   | `✗`    | `Colors.RED`    |
| `success` | `✓`    | `Colors.GREEN`  |

#### `saved(path: str, color: Colors = Colors.GREEN) -> None`

Symbol: `✓`

#### `new_best(**metrics: float | int | str, color: Colors = Colors.CYAN) -> None`

Symbol: `★`

---

### Lifecycle

#### `finish() -> None`

Prints summary with total elapsed time.

---

## Formatting Rules

### Metrics

| Type                                   | Format                  |
| -------------------------------------- | ----------------------- |
| `float` where `0.001 <= abs(v) < 1000` | `0.1234` (4 decimals)   |
| `float` otherwise                      | `1.00e-04` (scientific) |
| `int`                                  | as-is                   |
| `str`                                  | as-is                   |

### Time

Format: `HH:MM:SS`

---

## Internal State

| Field            | Type            | Purpose                      |
| ---------------- | --------------- | ---------------------------- |
| `_start_time`    | `float or None` | Set on first `epochs()` call |
| `_current_epoch` | `int`           | Current epoch index          |

---

## Output Format

### Header

```text
Training: {name}
================================================================================

```

### Epoch Start

```text
Epoch {n}/{total} [{elapsed}]
```

### Progress Bar

```text
train: 100%|████████████████████████████| 125/125 [32s, 3.91 batch/s]
val:   100%|████████████████████████████|  32/32  [4s, 8.00 batch/s]
```

Phase labels padded to 5 chars for alignment.

### Metrics

```text

train_loss: 0.8923  val_loss: 0.7654  lr: 1.00e-04

```

Blank line before and after.

### Status Messages

```text
> Starting training                    (cyan)
! Learning rate too high               (yellow)
✗ NaN detected                         (red)
✓ Model compiled                       (green)
✓ Saved: checkpoints/best.pt           (green)
★ New Best! val_loss: 0.5234           (cyan)
```

### Epoch Separator

```text
--------------------------------------------------------------------------------
```

### Finish

```text

================================================================================
Finished in 01:32:45
================================================================================
```

---

## Usage Example

```python
from mllog import TrainLogger, Colors

logger = TrainLogger(name="ResNet50", epochs=30)

best_loss = float("inf")

for epoch in logger.epochs():
    model.train()
    train_loss = 0.0

    for batch in logger.train(train_loader):
        loss = train_step(batch)
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    for batch in logger.val(val_loader):
        loss = val_step(batch)
        val_loss += loss.item()

    val_loss /= len(val_loader)

    logger.log(train_loss=train_loss, val_loss=val_loss)

    if val_loss < best_loss:
        logger.new_best(val_loss=val_loss, prev=best_loss)
        best_loss = val_loss
        save_model()
        logger.saved("checkpoints/best.pt")

logger.finish()
```

---

## Full Output Example

```text
Training: ResNet50
================================================================================

Epoch 1/30 [00:00:00]
train: 100%|████████████████████████████| 125/125 [32s, 3.91 batch/s]
val:   100%|████████████████████████████|  32/32  [4s, 8.00 batch/s]

train_loss: 0.8923  val_loss: 0.7654

★ New Best! val_loss: 0.7654  prev: inf
✓ Saved: checkpoints/best.pt
--------------------------------------------------------------------------------
Epoch 2/30 [00:00:36]
train: 100%|████████████████████████████| 125/125 [31s, 4.03 batch/s]
val:   100%|████████████████████████████|  32/32  [4s, 8.00 batch/s]

train_loss: 0.5102  val_loss: 0.5234

★ New Best! val_loss: 0.5234  prev: 0.7654
✓ Saved: checkpoints/best.pt
--------------------------------------------------------------------------------

================================================================================
Finished in 00:01:12
================================================================================
```

---

## File Structure

```text
mllog.py    # single file, all code
```
