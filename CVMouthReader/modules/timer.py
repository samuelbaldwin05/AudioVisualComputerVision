from functools import wraps
import time
from typing import Any, Callable, Tuple


def timer(func: Callable[..., Any]) -> Callable[..., Any]:
  """
  Decorator that measures the execution time of a function.

  Args:
    func (Callable[..., Any]): The function to be timed.

  Returns:
    Callable[..., Any]: The wrapped function.

  Example:
    @timer
    def my_function():
      # code to be timed
      pass

    my_function()  # prints the execution time of my_function
  """
  @wraps(func)
  def timeit_wrapper(*args: Tuple[Any], **kwargs: Any) -> Any:
    start_time = time.process_time_ns()
    result = func(*args, **kwargs)
    end_time = time.process_time_ns()
    total_time_ns = end_time - start_time
    total_time_ms = total_time_ns / float(1000000)
    print(f'Function {func.__name__}{args} {kwargs} Took {total_time_ns} ns OR {total_time_ms} ms')
    return result
  return timeit_wrapper