"""Regression tests for scalar/literal broadcasting in multi-input functions.

Several H3 functions take more than one Series argument and combine them with a
`zip` in Rust. Polars may pass a literal/scalar argument as a length-1 Series,
which -- without explicit broadcasting -- would silently truncate an ``N x 1``
call to a single row. These tests lock in the broadcasting contract now that the
functions are marked ``is_elementwise=True``.

Each positive test compares the broadcast form (literal argument) against the
materialized ``N x N`` baseline (the scalar repeated into a full column), and
asserts the output length equals the input length -- the regression that was
previously failing.
"""

import polars as pl
import pytest

import polars_h3 as plh3

# Resolution-10 origin and two of its ring-1 neighbours (distance 1).
ORIGIN = 622054503267303423
NEIGHBOR_A = 622054503267270655
NEIGHBOR_B = 622054503267237887
U64 = {"cell": pl.UInt64}


def _cells_df() -> pl.DataFrame:
    return pl.DataFrame(
        {"cell": [ORIGIN, NEIGHBOR_A, NEIGHBOR_B]},
        schema=U64,
    )


# --------------------------------------------------------------------------- #
# Two-cell functions: column of cells + a literal "other" cell.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "func",
    [
        plh3.grid_distance,
        plh3.are_neighbor_cells,
        plh3.cells_to_directed_edge,
        plh3.cell_to_local_ij,
        plh3.grid_path_cells,
    ],
)
def test_two_cell_broadcast_scalar_on_right(func):
    df = _cells_df()
    other = pl.lit(ORIGIN, dtype=pl.UInt64)

    broadcast = df.select(out=func("cell", other))["out"].to_list()
    baseline = (
        df.with_columns(other=pl.lit(ORIGIN, dtype=pl.UInt64))
        .select(out=func("cell", "other"))["out"]
        .to_list()
    )

    assert len(broadcast) == 3  # regression: was 1 before broadcasting
    assert broadcast == baseline


@pytest.mark.parametrize(
    "func",
    [
        plh3.grid_distance,
        plh3.are_neighbor_cells,
        plh3.cells_to_directed_edge,
        plh3.cell_to_local_ij,
        plh3.grid_path_cells,
    ],
)
def test_two_cell_broadcast_scalar_on_left(func):
    df = _cells_df()
    origin = pl.lit(ORIGIN, dtype=pl.UInt64)

    broadcast = df.select(out=func(origin, "cell"))["out"].to_list()
    baseline = (
        df.with_columns(origin=pl.lit(ORIGIN, dtype=pl.UInt64))
        .select(out=func("origin", "cell"))["out"]
        .to_list()
    )

    assert len(broadcast) == 3
    assert broadcast == baseline


def test_grid_distance_concrete_values():
    df = _cells_df()
    out = df.select(d=plh3.grid_distance("cell", pl.lit(ORIGIN, dtype=pl.UInt64)))
    # [origin, neighbour, neighbour] -> distances to origin
    assert out["d"].to_list() == [0, 1, 1]


def test_are_neighbor_concrete_values():
    df = _cells_df()
    out = df.select(n=plh3.are_neighbor_cells("cell", pl.lit(ORIGIN, dtype=pl.UInt64)))
    # a cell is not its own neighbour; the ring-1 cells are.
    assert out["n"].to_list() == [False, True, True]


# --------------------------------------------------------------------------- #
# grid_ring / grid_disk: broadcast either the cell or k.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("func", [plh3.grid_ring, plh3.grid_disk])
def test_ring_disk_scalar_k(func):
    df = _cells_df()
    # int k -> pl.lit(k), a length-1 literal broadcast across the cell column.
    broadcast = df.select(out=func("cell", 1))["out"].to_list()
    baseline = (
        df.with_columns(k=pl.lit(1, dtype=pl.Int32))
        .select(out=func("cell", "k"))["out"]
        .to_list()
    )
    assert len(broadcast) == 3
    assert broadcast == baseline


@pytest.mark.parametrize("func", [plh3.grid_ring, plh3.grid_disk])
def test_ring_disk_scalar_cell(func):
    df = pl.DataFrame({"k": [0, 1, 2]}, schema={"k": pl.Int32})
    cell = pl.lit(ORIGIN, dtype=pl.UInt64)

    broadcast = df.select(out=func(cell, "k"))["out"].to_list()
    baseline = (
        df.with_columns(cell=pl.lit(ORIGIN, dtype=pl.UInt64))
        .select(out=func("cell", "k"))["out"]
        .to_list()
    )
    assert len(broadcast) == 3  # regression: previously errored on length mismatch
    assert broadcast == baseline


# --------------------------------------------------------------------------- #
# local_ij_to_cell: origin column + scalar i / j.
# --------------------------------------------------------------------------- #
def test_local_ij_to_cell_scalar_ij():
    origin = 605034941285138431
    df = pl.DataFrame({"origin": [origin, origin, origin]}, schema={"origin": pl.UInt64})

    out = df.select(cell=plh3.local_ij_to_cell("origin", -123, -177))
    result = out["cell"].to_list()

    assert len(result) == 3  # regression: was 1 before broadcasting
    assert result == [origin, origin, origin]


# --------------------------------------------------------------------------- #
# Null propagation and length-mismatch errors.
# --------------------------------------------------------------------------- #
def test_broadcast_null_scalar_propagates():
    df = _cells_df()
    out = df.select(d=plh3.grid_distance("cell", pl.lit(None, dtype=pl.UInt64)))
    assert out["d"].to_list() == [None, None, None]


def test_mismatched_lengths_raise():
    left = pl.Series("a", [ORIGIN, NEIGHBOR_A], dtype=pl.UInt64)
    right = pl.Series("b", [ORIGIN, NEIGHBOR_A, NEIGHBOR_B], dtype=pl.UInt64)
    with pytest.raises(pl.exceptions.PolarsError):
        pl.select(plh3.grid_distance(left, right))
