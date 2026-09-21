"""Tests for brick_builder/envs/obs_spec.py: the tagged observation registry.

Every actor entry must name its real-hardware source and
that "a test fails if an actor entry has no source" -- this file is that test, plus the layout
and doc-table machinery every environment builds on.
"""

from __future__ import annotations

import numpy as np
import pytest

from brick_builder.envs import obs_spec


def test_every_registered_actor_entry_has_a_real_hardware_source():
    for name, entry in obs_spec.all_entries().items():
        if entry.scope != "actor":
            continue
        assert entry.source, f"{name} has no source"
        assert entry.source != "sim", f"{name} is actor-scoped but uses the privileged sim source"


def test_source_must_match_the_fixed_vocabulary():
    with pytest.raises(ValueError):
        obs_spec.ObsEntry("bogus", 3, "made_up_source", "actor")


def test_actor_entry_cannot_use_the_sim_source():
    with pytest.raises(ValueError):
        obs_spec.ObsEntry("bogus2", 1, "sim", "actor")


def test_register_is_idempotent_for_an_identical_redefinition():
    a = obs_spec.register(obs_spec.ObsEntry("dupe_test", 2, "fk", "actor"))
    b = obs_spec.register(obs_spec.ObsEntry("dupe_test", 2, "fk", "actor"))
    assert a == b


def test_register_rejects_a_conflicting_redefinition():
    obs_spec.register(obs_spec.ObsEntry("conflict_test", 2, "fk", "actor"))
    with pytest.raises(ValueError):
        obs_spec.register(obs_spec.ObsEntry("conflict_test", 3, "fk", "actor"))


def test_get_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        obs_spec.get("does_not_exist_anywhere")


def test_check_sources_rejects_a_critic_only_entry_in_an_actor_list():
    with pytest.raises(ValueError):
        obs_spec.check_sources(["read_stale"], scope="actor")
    # but a critic list may freely include actor-sourced entries (that is the whole point of an
    # asymmetric critic: it sees at least what the actor sees, plus privileged extras).
    obs_spec.check_sources(["joint_pos", "read_stale"], scope="critic")


def test_layout_sizes_respect_history():
    names = ["joint_pos", "cmd_grip"]  # history=4 and history=1 respectively
    layout = obs_spec.make_layout(names)
    assert layout.size == 6 * 4 + 1 * 1
    assert layout.slices["joint_pos"] == slice(0, 24)
    assert layout.slices["cmd_grip"] == slice(24, 25)


def test_make_layout_rejects_a_repeated_name():
    with pytest.raises(ValueError):
        obs_spec.make_layout(["joint_pos", "joint_pos"])


def test_make_space_matches_layout_size():
    names = list(obs_spec.PROPRIO_ENTRIES)
    space = obs_spec.make_space(names)
    assert space.shape == (obs_spec.make_layout(names).size,)


def test_pack_round_trips_named_values():
    layout = obs_spec.make_layout(["cmd_grip", "cmd_compliance"])
    values = {"cmd_grip": np.array([0.4]), "cmd_compliance": np.array([0.7, 0.9])}
    packed = layout.pack(values)
    assert packed.dtype == np.float32
    np.testing.assert_allclose(packed, [0.4, 0.7, 0.9], atol=1e-6)


def test_pack_rejects_wrong_sized_value():
    layout = obs_spec.make_layout(["cmd_grip"])
    with pytest.raises(ValueError):
        layout.pack({"cmd_grip": np.array([0.1, 0.2])})


def test_markdown_table_lists_every_name_once():
    names = list(obs_spec.PROPRIO_ENTRIES)
    table = obs_spec.markdown_table(names)
    for name in names:
        assert f"`{name}`" in table
    lines = [l for l in table.splitlines() if l.startswith("| `")]
    assert len(lines) == len(names)


def test_critic_only_entries_are_all_sim_sourced():
    for name in obs_spec.CRITIC_ONLY_ENTRIES:
        entry = obs_spec.get(name)
        assert entry.scope == "critic"
        assert entry.source == "sim"
