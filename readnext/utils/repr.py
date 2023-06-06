import polars as pl


def generate_frame_repr(frame: pl.DataFrame) -> str:
    """Generate a string representation of a `pl.DataFrame`."""
    return f"[pl.DataFrame, shape={frame.shape}, columns={frame.columns}]"
