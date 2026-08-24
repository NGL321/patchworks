"""The arena, the arm, the pucks and the zones, against the spec's tables.

Every number here is transcribed from `docs/spec/03-the-sandbox.md`, which got
them by building the thing and watching it. These tests exist so that a number
cannot drift out of the arena without something saying so.
"""

import mujoco
import numpy as np
import pytest

from patchworks.sandbox import (
    ARENA_XML,
    HELDOUT_SECTOR,
    N_PUCKS,
    N_ZONES,
    ZONE_RADIUS,
    ZONE_XY,
    in_heldout_sector,
)

G = 9.81  # table friction is modelled as joint frictionloss, mu*m*g

LINK_LENGTH = (0.20, 0.16, 0.10)
TORQUE_LIMIT = (3.0, 2.0, 1.0)
JOINT_RANGE = (np.pi, 2.6, 2.6)
DAMPING = (0.8, 0.5, 0.3)
ARMATURE = (0.12, 0.02, 0.002)

PUCK_RADIUS = (0.035, 0.045, 0.055)
PUCK_MASS = (0.05, 0.10, 0.20)
PUCK_MU = (0.20, 0.30, 0.45)
# mu*m*g, written into the arena to three significant figures
PUCK_FRICTIONLOSS = (0.098, 0.294, 0.883)
ECCENTRICITY = 0.018  # puck 1's centre of mass, off the geometric centre

RING_WALL_RADIUS = 0.52
PEDESTAL_RADIUS = 0.08
PADDLE_RADIUS = 0.03
ARM_REACH = 0.46  # to the tip site
ARM_EXTENT = 0.49  # to the outside of the paddle
ZONE_ANGLES_DEG = (90.0, 210.0, 330.0)


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(ARENA_XML)


def _gid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _jid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


# -- planar by construction -----------------------------------------------------


def test_the_world_is_planar_by_construction(model):
    """Gravity is zero and every joint is a hinge about z or a slide in x/y.

    Not a floor-and-gravity trick: going 3D is adding joints and turning
    gravity on, which is removing constraints rather than porting an engine.
    """
    assert np.all(model.opt.gravity == 0.0)
    for j in range(model.njnt):
        axis = model.jnt_axis[j]
        kind = model.jnt_type[j]
        if kind == mujoco.mjtJoint.mjJNT_HINGE:
            assert np.allclose(axis, [0, 0, 1])
        elif kind == mujoco.mjtJoint.mjJNT_SLIDE:
            assert np.allclose(axis, [1, 0, 0]) or np.allclose(axis, [0, 1, 0])
        else:
            pytest.fail(f"joint {j} is neither a hinge nor a slide")


def test_table_friction_is_joint_frictionloss_not_a_contact(model):
    """No supporting surface: the backdrop geom does not collide with anything."""
    table = _gid(model, "table")
    assert model.geom_contype[table] == 0
    assert model.geom_conaffinity[table] == 0


# -- the body -------------------------------------------------------------------


def test_the_arm_matches_the_spec_table(model):
    lengths = (
        model.body_pos[_bid(model, "link1")][0],
        model.body_pos[_bid(model, "link2")][0],
        model.site_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")][0],
    )
    assert lengths == pytest.approx(LINK_LENGTH)
    assert sum(LINK_LENGTH) == pytest.approx(ARM_REACH)

    for i, name in enumerate(("j0", "j1", "j2")):
        dof = model.jnt_dofadr[_jid(model, name)]
        assert model.actuator_ctrlrange[i][1] == pytest.approx(TORQUE_LIMIT[i])
        assert model.actuator_ctrlrange[i][0] == pytest.approx(-TORQUE_LIMIT[i])
        assert model.jnt_range[_jid(model, name)] == pytest.approx(
            (-JOINT_RANGE[i], JOINT_RANGE[i]), abs=1e-4
        )
        assert model.dof_damping[dof] == pytest.approx(DAMPING[i])
        assert model.dof_armature[dof] == pytest.approx(ARMATURE[i])


def test_the_armature_ladder_is_uneven_and_wide(model):
    """The rungs are deliberately uneven, so the ladder cannot be read off as
    three joint timescales matching three levels of the core."""
    j0, j1, j2 = ARMATURE
    assert j0 / j1 == pytest.approx(6.0)
    assert j1 / j2 == pytest.approx(10.0)


def test_the_paddle_is_the_arm_s_outer_extent(model):
    tip = _gid(model, "g_tip")
    assert model.geom_size[tip][0] == pytest.approx(PADDLE_RADIUS)
    assert ARM_REACH + PADDLE_RADIUS == pytest.approx(ARM_EXTENT)


def test_the_pedestal_is_a_real_obstacle_the_arm_collides_with(model):
    """It walls off the shoulder, where the arm is singular, and makes the
    paddle's reachable set an annulus rather than a disk. Only link0 is
    excluded from it -- link1 and link2 must collide, or the arm folds through."""
    pedestal = _gid(model, "g_pedestal")
    assert model.geom_size[pedestal][0] == pytest.approx(PEDESTAL_RADIUS)
    assert model.geom_contype[pedestal] and model.geom_conaffinity[pedestal]

    # MuJoCo does not filter the base/link0 contact pair, because the base is
    # welded to the world and that disables the parent filter. Without the
    # explicit exclusion link0 lives permanently inside the pedestal.
    base, link0 = _bid(model, "base"), _bid(model, "link0")
    assert (base << 16) + link0 in model.exclude_signature.tolist()


# -- the world ------------------------------------------------------------------


def test_the_ring_wall_sits_outside_the_arm_s_extent(model):
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    radii = [
        float(np.linalg.norm(model.geom_pos[g][:2]))
        for g, name in enumerate(names)
        if name is not None and name.startswith("wall_")
    ]
    assert radii
    assert radii == pytest.approx([RING_WALL_RADIUS] * len(radii), abs=1e-3)
    assert RING_WALL_RADIUS > ARM_EXTENT


def test_the_pucks_match_the_spec_table(model):
    for i in range(N_PUCKS):
        disc = _gid(model, f"g_puck_{i}")
        assert model.geom_size[disc][0] == pytest.approx(PUCK_RADIUS[i])

        # The orientation marker is a non-colliding visual geom; it carries the
        # default density, so it adds ~0.3 g to a puck whose spec mass is a
        # round number. Under a percent, and it is what every measurement in
        # the spec was taken on.
        body = _bid(model, f"puck_{i}")
        assert model.body_mass[body] == pytest.approx(PUCK_MASS[i], rel=0.01)

        # mu differs per puck, not just its product with mass: a free puck
        # decelerates at mu*g, so this is what makes the difference survive the
        # coasting phase.
        dof = model.jnt_dofadr[_jid(model, f"p{i}_x")]
        assert model.dof_frictionloss[dof] == pytest.approx(PUCK_FRICTIONLOSS[i])
        mu = model.dof_frictionloss[dof] / (PUCK_MASS[i] * G)
        assert mu == pytest.approx(PUCK_MU[i], abs=5e-4)
        assert model.dof_frictionloss[dof] == pytest.approx(
            model.dof_frictionloss[dof + 1]
        ), "x and y frictionloss must match, or the table is anisotropic by axis"


def test_only_puck_one_is_eccentric(model):
    """Puck 1's centre of mass is off-centre, so a contact through the rim
    exerts a torque about a point that moves with theta and theta feeds back
    into where the puck goes. The other two are central."""
    offsets = [float(np.linalg.norm(model.body_ipos[_bid(model, f"puck_{i}")])) for i in range(3)]
    assert offsets[1] == pytest.approx(ECCENTRICITY, abs=5e-4)
    assert offsets[0] < 5e-4
    assert offsets[2] < 5e-4


def test_the_eccentric_mass_is_invisible_in_the_render(model):
    """It was chosen over a non-circular puck precisely because it does not
    touch the render: a rotating silhouette would partly un-hide the variable
    it was introduced to hide."""
    cm = _gid(model, "g_puck_1_cm")
    disc = _gid(model, "g_puck_1")
    assert model.geom_contype[cm] == 0 and model.geom_conaffinity[cm] == 0
    # buried inside the solid cylinder, and the same colour as it
    assert np.linalg.norm(model.geom_pos[cm][:2]) + model.geom_size[cm][0] < (
        model.geom_size[disc][0]
    )
    assert np.allclose(model.geom_rgba[cm], model.geom_rgba[disc])


def test_sub_threshold_holds_are_stiff(model):
    """A frictionloss constraint's position residual is identically zero, so it
    takes its impedance from solimp[0]. At the default 0.9 about a tenth of the
    load leaks through and a held puck creeps."""
    for i in range(N_PUCKS):
        dof = model.jnt_dofadr[_jid(model, f"p{i}_x")]
        for k in range(3):
            assert model.dof_solimp[dof + k][0] == pytest.approx(0.9999)


def test_the_zones_match_the_spec(model):
    for z in range(N_ZONES):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"zone_{z}")
        pos = model.site_pos[sid][:2]
        assert model.site_size[sid][0] == pytest.approx(ZONE_RADIUS)
        assert pos == pytest.approx(ZONE_XY[z])
        assert float(np.linalg.norm(pos)) == pytest.approx(0.30, abs=1e-3)
        angle = np.rad2deg(np.arctan2(pos[1], pos[0])) % 360.0
        assert angle == pytest.approx(ZONE_ANGLES_DEG[z], abs=0.5)


def test_no_zone_centre_lies_inside_the_held_out_wedge(model):
    """The sector holdout withholds a target-puck position; it must not also
    withhold a place to push a puck *to*, or the two axes couple.

    No zone centre is in the wedge, and zone 0 -- the only one anywhere near it
    -- is tangent to the wedge's upper edge with about half a degree to spare,
    so the holdout does not silently restrict approach geometry for
    pair-holdout tasks either. The margin is read off the wedge constant rather
    than a transcribed 75 degrees, because widening the wedge is exactly the
    change the spec says would couple the axes with nothing announcing it."""
    upper_edge = np.rad2deg(HELDOUT_SECTOR[1])
    for z in range(N_ZONES):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"zone_{z}")
        assert not in_heldout_sector(model.site_pos[sid][:2])

    half_width = np.rad2deg(np.arcsin(ZONE_RADIUS / 0.30))
    clearance = (ZONE_ANGLES_DEG[0] - half_width) - upper_edge
    assert 0.0 < clearance < 1.0


def test_the_control_rate_is_fifty_hertz(model):
    assert model.opt.timestep == pytest.approx(0.002)


def test_the_touch_sensors_are_the_whole_sensor_block(model):
    """The observation resolves them by name, but a fourth sensor would still
    be a fourth number nobody declared a home for."""
    assert model.nsensor == 3
    for i in range(3):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"t{i}")
        assert sid != -1
        assert model.sensor_type[sid] == mujoco.mjtSensor.mjSENS_TOUCH
        assert model.sensor_dim[sid] == 1
