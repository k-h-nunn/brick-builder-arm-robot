"""Tests for `python -m brick_builder.language` (`brick_builder/language/__main__.py`).

These call `main()` directly (rather than spawning a subprocess) and capture stdout with
`capsys`, which is faster and keeps the test in the same coverage/interpreter session; the
module is a thin, mostly-print wrapper over `compile_command`, already tested in depth in
`test_compiler.py`.
"""

from __future__ import annotations

from brick_builder.language.__main__ import main, render_goal
from brick_builder.language.compiler import compile_command


def test_main_prints_paraphrase_placement_text_and_picture(capsys):
    exit_code = main(["put", "the", "red", "2x4", "at", "column", "3", "row", "2"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "paraphrase:" in out
    assert "2x4 (2,1,0) red" in out
    assert "layer 1 (z=0)" in out
    assert " R " in out or " R\n" in out  # the red brick's colour code somewhere in the picture
    assert "key:" in out


def test_main_accepts_the_command_as_a_single_argument():
    """`argv` can be one already-joined string, or several words - `nargs="+"` joins them the
    same way either way, matching `python -m brick_builder.language "put the red 2x4 ..."`."""

    assert compile_command("put the red 2x4 at column 3 row 2").understood


def test_main_reports_problems_and_a_nonzero_exit_when_not_understood(capsys):
    exit_code = main(["put", "the", "purple", "2x2", "at", "column", "1", "row", "1"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "command not understood" in out
    assert "problems:" in out
    assert "purple" in out


def test_main_multi_step_command_shows_more_than_one_layer(capsys):
    exit_code = main([
        "put", "a", "red", "2x2", "at", "column", "1", "row", "1", "then",
        "put", "a", "blue", "2x2", "on", "top", "of", "the", "red", "brick",
    ])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "layer 1 (z=0)" in out
    assert "layer 2 (z=1)" in out


def test_main_grid_option_loads_current_grid_from_a_file(tmp_path, capsys):
    grid_file = tmp_path / "grid.txt"
    grid_file.write_text("2x2 (0,0,0) red\n")

    exit_code = main(["--grid", str(grid_file), "put", "a", "blue", "2x2", "on", "top", "of", "the", "red", "brick"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "2x2 (0,0,1) blue" in out


def test_main_inventory_option_reports_a_shortage(tmp_path, capsys):
    inventory_file = tmp_path / "inventory.txt"
    inventory_file.write_text("# comment lines are ignored\n0 red 2x4\n")

    exit_code = main(["--inventory", str(inventory_file), "put", "the", "red", "2x4", "at", "column", "1", "row", "1"])
    out = capsys.readouterr().out

    assert exit_code == 1
    assert "available bricks are 0 red 2x4" in out


def test_main_inventory_option_allows_a_stocked_brick(tmp_path, capsys):
    inventory_file = tmp_path / "inventory.txt"
    inventory_file.write_text("2 red 2x4\n")

    exit_code = main(["--inventory", str(inventory_file), "put", "the", "red", "2x4", "at", "column", "1", "row", "1"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "2x4 (0,0,0) red" in out


def test_render_goal_reports_nothing_to_place_for_an_empty_goal():
    from brick_builder.language.goal import Goal

    assert render_goal(Goal(steps=())) == "(nothing to place)"


def test_render_goal_labels_columns_and_rows_from_one():
    result = compile_command("put a red 2x2 at column 1 row 1")
    picture = render_goal(result.goal)
    header = picture.splitlines()[1]
    assert header.split()[:3] == ["1", "2", "3"]
