"""Self-contained MJCF test scenes for tests/render (no dependency on other packages).

Exposed as fixtures, not a plain importable module, because pytest's
`--import-mode=importlib` (see pyproject.toml) does not add test directories to
`sys.path`, so a sibling test file cannot `import scenes` reliably; a colocated
conftest.py is always discovered regardless of import mode.
"""

import pytest

# A camera looking straight down (+z pos, identity quat -> local -z is world -z) at a
# grey table with a red and a blue box on it. The table plane is large enough that the
# camera's whole field of view stays inside it, so there is no true background pixel.
TWO_BOX_MJCF = """
<mujoco model="two_box">
  <visual>
    <global offwidth="512" offheight="512"/>
    <!-- offsamples=0 disables multisample anti-aliasing: with it left at the
         default, GPU-driver edge blending is not bit-reproducible frame to
         frame, which breaks the "same state renders identical bytes" contract. -->
    <quality offsamples="0"/>
    <map znear="0.01" zfar="5"/>
  </visual>
  <asset>
    <material name="mat_red" rgba="0.8 0.1 0.1 1"/>
    <material name="mat_blue" rgba="0.1 0.1 0.8 1"/>
    <material name="mat_table" rgba="0.5 0.5 0.5 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" size="1 1 0.01" pos="0 0 0" material="mat_table"/>
    <body name="box_red" pos="-0.1 0 0.05">
      <freejoint/>
      <geom name="box_red_geom" type="box" size="0.05 0.05 0.05" material="mat_red"/>
    </body>
    <body name="box_blue" pos="0.1 0 0.05">
      <freejoint/>
      <geom name="box_blue_geom" type="box" size="0.05 0.05 0.05" material="mat_blue"/>
    </body>
    <camera name="cam0" pos="0 0 1" quat="1 0 0 0" fovy="45"/>
  </worldbody>
</mujoco>
"""

# Same scene but with a small table, so the sky shows around its edges (true background,
# instance == -1) and a collision-only box (group 3) that must never appear in a render.
TWO_BOX_SMALL_TABLE_MJCF = TWO_BOX_MJCF.replace('size="1 1 0.01"', 'size="0.08 0.08 0.01"').replace(
    "</worldbody>",
    '    <body name="hidden_collider" pos="0 0 0.3">\n'
    '      <geom name="hidden_collider_geom" type="box" size="0.2 0.2 0.2" group="3" rgba="0 1 0 1"/>\n'
    "    </body>\n"
    "  </worldbody>",
)

# A single flat plane, camera pointed straight down from a known height, for the depth
# accuracy test: every visible pixel should read back as (camera height - plane height).
PLANE_MJCF = """
<mujoco model="plane">
  <visual>
    <global offwidth="256" offheight="256"/>
    <quality offsamples="0"/>
    <map znear="0.01" zfar="5"/>
  </visual>
  <asset>
    <material name="mat_table" rgba="0.6 0.6 0.6 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="plane" size="1 1 0.01" pos="0 0 0" material="mat_table"/>
    <camera name="cam0" pos="0 0 0.75" quat="1 0 0 0" fovy="30"/>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def two_box_mjcf() -> str:
    return TWO_BOX_MJCF


@pytest.fixture
def two_box_small_table_mjcf() -> str:
    return TWO_BOX_SMALL_TABLE_MJCF


@pytest.fixture
def plane_mjcf() -> str:
    return PLANE_MJCF
