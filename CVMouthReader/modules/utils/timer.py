from functools import wraps
import time
from typing import Any, Callable, Tuple

# Hardcoded log file name
LOG_FILE = "CVMouthReader/data/output/execution_times.txt"

def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that measures the execution time of a function and logs it to a hardcoded file.

    Args:
        func (Callable[..., Any]): The function to be timed.

    Returns:
        Callable[..., Any]: The wrapped function.
    """
    @wraps(func)
    def timeit_wrapper(*args: Tuple[Any], **kwargs: Any) -> Any:
        start_time = time.process_time_ns()
        result = func(*args, **kwargs)
        end_time = time.process_time_ns()
        total_time_ns = end_time - start_time
        total_seconds = total_time_ns / 1e9
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        formatted_time = f"{minutes}:{seconds:.2f}"
        log_message = f'Function {func.__name__}{args} {kwargs} Took {total_time_ns} ns. {formatted_time}\n'

        # Print to console
        print(log_message, end="")

        # Write to hardcoded log file
        with open(LOG_FILE, "a") as f:
            f.write(log_message)

        return result
    return timeit_wrapper
