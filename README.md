# ML Simple Logging

Minimal Python logging library for machine learning training loops. Wraps tqdm for progress bars, provides colored status messages, and controls the training loop.

## Installation

```bash
pip install ml-simple-logging
```

## Quick Start

```python
from ml_simple_logging import TrainLogger

logger = TrainLogger(name="MyModel", epochs=10)

for epoch in logger.epochs():
    for batch in logger.train(train_loader):
        # Training step
        pass
    
    for batch in logger.val(val_loader):
        # Validation step
        pass
    
    logger.log(loss=0.5234, accuracy=0.98)

logger.finish()
```

## Configuration

### TrainLogger Parameters

```python
TrainLogger(
    name: str = "Training",     # Display name for training run
    epochs: int = 1,             # Total epoch count
    bar_width: int = 40,        # Progress bar character width
)
```

## Use Cases

### Basic Training Loop

```python
from ml_simple_logging import TrainLogger

logger = TrainLogger(name="ResNet50", epochs=30)

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

logger.finish()
```

### Model Checkpointing

```python
from ml_simple_logging import TrainLogger

logger = TrainLogger(name="MyModel", epochs=100)
best_loss = float("inf")

for epoch in logger.epochs():
    train_loss = train()
    val_loss = validate()
    
    logger.log(train_loss=train_loss, val_loss=val_loss)
    
    if val_loss < best_loss:
        logger.new_best(val_loss=val_loss, prev=best_loss)
        best_loss = val_loss
        torch.save(model.state_dict(), "best.pt")
        logger.saved("best.pt")

logger.finish()
```

### Training with Warnings and Errors

```python
from ml_simple_logging import TrainLogger

logger = TrainLogger(name="MyModel", epochs=10)

logger.info("Starting training...")
logger.success("Model compiled successfully")

for epoch in logger.epochs():
    loss = train()
    
    if torch.isnan(loss):
        logger.error("NaN loss detected!")
        break
    
    if loss > 1.0:
        logger.warn("High loss detected, consider lowering learning rate")
    
    logger.log(loss=loss, lr=get_lr())

logger.finish()
```

### Multi-Metric Logging

```python
from ml_simple_logging import TrainLogger

logger = TrainLogger(name="MyModel", epochs=20)

for epoch in logger.epochs():
    train()
    
    metrics = evaluate()
    logger.log(
        loss=metrics["loss"],
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        lr=current_lr,
    )

logger.finish()
```

### Fine-Tuning Pre-trained Model

```python
from ml_simple_logging import TrainLogger

logger = TrainLogger(name="Fine-tune BERT", epochs=5)

logger.info("Loading pre-trained weights...")
load_weights()

for epoch in logger.epochs():
    train_loss = fine_tune()
    val_acc = evaluate()
    
    logger.log(train_loss=train_loss, val_acc=val_acc)
    
    if val_acc > 0.95:
        logger.success(f"Target accuracy reached: {val_acc:.2%}")
        break

logger.finish()
```

## API Reference

### Status Messages

```python
logger.info(message)       # > message (cyan)
logger.warn(message)       # ! message (yellow)
logger.error(message)      # ✗ message (red)
logger.success(message)    # ✓ message (green)
logger.saved(path)         # ✓ Saved: path (green)
logger.new_best(**metrics) # ★ New Best! metrics (cyan)
```

### Progress Bars

```python
logger.train(iterable)  # Wraps iterable with "train" progress bar
logger.val(iterable)    # Wraps iterable with "val" progress bar
```

## Output Example

```
Training: ResNet50
========================================

Epoch 1/30 [00:00:00]
train: 100%|████████████████████████| 125/125 [32s, 3.91 batch/s]
val:   100%|████████████████████████|  32/32  [4s, 8.00 batch/s]

train_loss: 0.8923  val_loss: 0.7654

★ New Best! val_loss: 0.7654  prev: inf
✓ Saved: checkpoints/best.pt
----------------------------------------
Epoch 2/30 [00:00:36]
train: 100%|████████████████████████| 125/125 [31s, 4.03 batch/s]
val:   100%|████████████████████████|  32/32  [4s, 8.00 batch/s]

train_loss: 0.5102  val_loss: 0.5234

★ New Best! val_loss: 0.5234  prev: 0.7654
✓ Saved: checkpoints/best.pt
----------------------------------------

========================================
Finished in 00:01:12
========================================
```

## Requirements

- Python >= 3.11
- tqdm
