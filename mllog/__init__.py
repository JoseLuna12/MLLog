from typing import Iterator, Iterable, TypeVar, cast
import shutil
import time
from tqdm import tqdm

T = TypeVar("T")


class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_value(v: float | int | str) -> str:
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if 0.001 <= abs(v) < 1000:
            return f"{v:.4f}"
        else:
            return f"{v:.2e}"
    return str(v)


class TrainLogger:
    def __init__(
        self, name: str = "Training", epochs: int = 1, bar_width: int | None = None
    ):
        self.name = name
        self.epochs_total = epochs
        self._bar_ncols: int | None = bar_width  # None => tqdm uses terminal width
        self.bar_width = (
            bar_width if bar_width is not None else self._get_terminal_width()
        )
        self._start_time: float | None = None
        self._current_epoch: int = 0

    def _get_terminal_width(self) -> int:
        # tqdm uses ncols=None to auto-size; we mirror that width for dividers
        try:
            return shutil.get_terminal_size().columns
        except OSError:
            return 40

    def _print_header(self) -> None:
        print(f"Training: {self.name}")
        print("=" * self.bar_width)
        print()

    def _print_epoch_start(self, epoch: int) -> None:
        if self._start_time is None:
            self._start_time = time.time()
        elapsed = time.time() - self._start_time
        print(f"Epoch {epoch}/{self.epochs_total} [{format_time(elapsed)}]")

    def _print_epoch_separator(self) -> None:
        print("-" * self.bar_width)

    def epochs(self) -> Iterator[int]:
        if self._current_epoch == 0:
            self._print_header()
        for epoch in range(1, self.epochs_total + 1):
            self._current_epoch = epoch
            self._print_epoch_start(epoch)
            yield epoch
            if epoch < self.epochs_total:
                self._print_epoch_separator()
        print()

    def train(self, iterable: Iterable[T]) -> Iterator[T]:
        return cast(
            Iterator[T], tqdm(iterable, desc="train", ncols=self._bar_ncols, leave=False)
        )

    def val(self, iterable: Iterable[T]) -> Iterator[T]:
        return cast(
            Iterator[T], tqdm(iterable, desc="val  ", ncols=self._bar_ncols, leave=False)
        )

    def log(self, **metrics: float | int | str) -> None:
        formatted = [f"{k}: {format_value(v)}" for k, v in metrics.items()]
        print()
        print("  ".join(formatted))
        print()

    def _print_status(
        self, symbol: str, message: str, color: str = Colors.CYAN
    ) -> None:
        print(f"{color}{symbol} {message}{Colors.RESET}")

    def info(self, message: str, color: str = Colors.CYAN) -> None:
        self._print_status(">", message, color)

    def warn(self, message: str, color: str = Colors.YELLOW) -> None:
        self._print_status("!", message, color)

    def error(self, message: str, color: str = Colors.RED) -> None:
        self._print_status("✗", message, color)

    def success(self, message: str, color: str = Colors.GREEN) -> None:
        self._print_status("✓", message, color)

    def saved(self, path: str, color: str = Colors.GREEN) -> None:
        self._print_status("✓", f"Saved: {path}", color)

    def new_best(self, color: str = Colors.CYAN, **metrics: float | int | str) -> None:
        formatted = [f"{k}: {format_value(v)}" for k, v in metrics.items()]
        self._print_status("★", f"New Best! {'  '.join(formatted)}", color)

    def finish(self) -> None:
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            print()
            print("=" * self.bar_width)
            print(f"Finished in {format_time(elapsed)}")
            print("=" * self.bar_width)
