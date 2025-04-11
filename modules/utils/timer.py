from functools import wraps
import time
from typing import Any, Callable, Tuple

# Hardcoded log file name
LOG_FILE = "data/output/execution_times.txt"

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
        # Start timing
        start_time = time.time_ns()  # Use time_ns() for wall-clock time
        result = func(*args, **kwargs)
        # End timing
        end_time = time.time_ns()
        total_time_ns = end_time - start_time

        # Convert nanoseconds to minutes and seconds
        total_seconds = total_time_ns / 1e9
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60

        # Format minutes and seconds with leading zeros
        formatted_minutes = f"{minutes:02d}"  # Ensures 2 digits (e.g., 05)
        formatted_seconds = f"{seconds:06.3f}"  # Ensures 6 characters with 3 decimal places (e.g., 09.123)

        # Format the total time as MM:SS.sss
        formatted_time = f"{formatted_minutes}:{formatted_seconds}"

        # Log message
        log_message = (
            f'Function {func.__name__}{args}{kwargs} '
            f'Took {total_time_ns} ns ({formatted_time})\n'
        )

        # Print to console
        print(log_message, end="")

        # Write to hardcoded log file
        with open(LOG_FILE, "a") as f:
            f.write(log_message)

        return result
    return timeit_wrapper