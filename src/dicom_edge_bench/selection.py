"""Frame subset parsing, kept dependency-free for easy testing."""


def parse_selection(spec: str, total: int) -> list[int]:
    """Accept 'all', 'first:N', 'uniform:N', 'start:stop[:step]', or '1,3,5'."""
    if total < 1:
        raise ValueError("DICOM must contain at least one frame")
    spec = spec.strip().lower()
    if spec == "all":
        result = list(range(total))
    elif spec.startswith("first:"):
        n = int(spec.split(":", 1)[1])
        if not 1 <= n <= total:
            raise ValueError("first:N requires 1 <= N <= frame count")
        result = list(range(n))
    elif spec.startswith("uniform:"):
        n = int(spec.split(":", 1)[1])
        if not 1 <= n <= total:
            raise ValueError("uniform:N requires 1 <= N <= frame count")
        # Integer division avoids NumPy as a dependency for CLI parsing.
        result = [(i * (total - 1)) // (n - 1) for i in range(n)] if n > 1 else [0]
    elif ":" in spec:
        components = spec.split(":")
        if len(components) not in (2, 3):
            raise ValueError("use start:stop[:step]")
        start = int(components[0]) if components[0] else 0
        stop = int(components[1]) if components[1] else total
        step = int(components[2]) if len(components) == 3 else 1
        if step <= 0:
            raise ValueError("step must be positive")
        result = list(range(start, stop, step))
    else:
        result = [int(i.strip()) for i in spec.split(",")]
    if not result:
        raise ValueError("selection must not be empty")
    if any(i < 0 or i >= total for i in result):
        raise ValueError(f"selection indices must be in [0, {total - 1}]")
    return result


def chunks(items: list[int], size: int):
    if size < 1:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]
