from mllog import TrainLogger, Colors, format_time, format_value


def test_colors_values():
    assert Colors.GREEN == "\033[92m"
    assert Colors.YELLOW == "\033[93m"
    assert Colors.RED == "\033[91m"
    assert Colors.CYAN == "\033[96m"
    assert Colors.RESET == "\033[0m"


def test_format_time():
    assert format_time(0) == "00:00:00"
    assert format_time(3665) == "01:01:05"
    assert format_time(60) == "00:01:00"
    assert format_time(3600) == "01:00:00"


def test_format_value_int():
    assert format_value(42) == "42"
    assert format_value(0) == "0"


def test_format_value_float():
    assert format_value(0.5234) == "0.5234"
    assert format_value(1000.0) == "1.00e+03"
    assert format_value(0.0001) == "1.00e-04"


def test_format_value_str():
    assert format_value("hello") == "hello"
    assert format_value("test") == "test"


def test_train_logger_initialization():
    logger = TrainLogger(name="Test", epochs=10, bar_width=50)
    assert logger.name == "Test"
    assert logger.epochs_total == 10
    assert logger.bar_width == 50
    assert logger._start_time is None
    assert logger._current_epoch == 0


def test_train_logger_default_params():
    logger = TrainLogger()
    assert logger.name == "Training"
    assert logger.epochs_total == 1
    assert logger.bar_width == 40


def test_train_logger_epochs_count():
    logger = TrainLogger(epochs=3)
    epochs_list = list(logger.epochs())
    assert len(epochs_list) == 3
    assert epochs_list == [1, 2, 3]
    assert logger._start_time is not None
    assert logger._current_epoch == 3


def test_train_logger_start_time_set():
    logger = TrainLogger(epochs=1)
    assert logger._start_time is None
    list(logger.epochs())
    assert logger._start_time is not None


def test_train_logger_train_iterator():
    logger = TrainLogger(epochs=1)
    data = [1, 2, 3, 4, 5]
    result = list(logger.train(data))
    assert result == data


def test_train_logger_val_iterator():
    logger = TrainLogger(epochs=1)
    data = [1, 2, 3, 4, 5]
    result = list(logger.val(data))
    assert result == data


def test_train_logger_log_metrics():
    logger = TrainLogger(epochs=1)
    logger.log(loss=0.5234, accuracy=98, lr=1e-4)


def test_train_logger_info():
    logger = TrainLogger(epochs=1)
    logger.info("Test info message")


def test_train_logger_warn():
    logger = TrainLogger(epochs=1)
    logger.warn("Test warning message")


def test_train_logger_error():
    logger = TrainLogger(epochs=1)
    logger.error("Test error message")


def test_train_logger_success():
    logger = TrainLogger(epochs=1)
    logger.success("Test success message")


def test_train_logger_saved():
    logger = TrainLogger(epochs=1)
    logger.saved("checkpoints/test.pt")


def test_train_logger_new_best():
    logger = TrainLogger(epochs=1)
    logger.new_best(loss=0.5234, prev=0.6)


def test_train_logger_finish():
    logger = TrainLogger(epochs=1)
    list(logger.epochs())
    logger.finish()
    assert logger._start_time is not None


def test_multiple_epochs():
    logger = TrainLogger(epochs=2)
    for epoch in logger.epochs():
        logger.log(epoch=epoch)
    logger.finish()
