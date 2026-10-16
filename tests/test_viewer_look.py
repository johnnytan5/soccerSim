import numpy as np
import pytest

from soccer_sim.config import SimConfig
from soccer_sim.field import build_world
from soccer_sim.viewer_look import COLLISION_GROUP, TEAM_RGBA, VISUAL_GROUP, apply_look

HOME, AWAY = TEAM_RGBA["home"], TEAM_RGBA["away"]


@pytest.fixture
def world():
    m = build_world(SimConfig(), 2)
    root = m.body_rootid[m.geom_bodyid]
    robots = sorted(set(root[(m.geom_group == COLLISION_GROUP)].tolist()))
    assert len(robots) == 2
    return m, root, robots


def test_simple_bodies_tint_each_robots_collision_shapes_by_team_and_hide_the_meshes(world):
    m, root, (r0, r1) = world
    others_before = m.geom_rgba[(m.geom_group != COLLISION_GROUP)].copy()
    groups = apply_look(m, {r0: HOME, r1: AWAY})
    assert groups == {VISUAL_GROUP: 0, COLLISION_GROUP: 1}
    for r, color in ((r0, HOME), (r1, AWAY)):
        sel = (root == r) & (m.geom_group == COLLISION_GROUP)
        assert sel.sum() == 21
        assert np.allclose(m.geom_rgba[sel], color, atol=1e-6) and (m.geom_matid[sel] == -1).all()
    assert np.array_equal(m.geom_rgba[(m.geom_group != COLLISION_GROUP)], others_before)  # meshes, ball, posts: untouched
    assert not m.light_castshadow.any()


def test_detailed_bodies_keep_meshes_and_tint_only_the_light_grey_parts(world):
    m, root, (r0, r1) = world
    dark = (m.geom_group == VISUAL_GROUP) & np.isclose(m.geom_rgba[:, 0], 0.21, atol=0.02)
    dark_before = m.geom_rgba[dark].copy()
    collision_before = m.geom_rgba[m.geom_group == COLLISION_GROUP].copy()
    groups = apply_look(m, {r0: HOME, r1: AWAY}, detailed=True)
    assert groups == {VISUAL_GROUP: 1, COLLISION_GROUP: 0}
    for r, color in ((r0, HOME), (r1, AWAY)):
        sel = (root == r) & (m.geom_group == VISUAL_GROUP) & np.isclose(m.geom_rgba[:, 0], color[0])
        assert sel.sum() == 28
    assert dark.sum() == 10 and np.array_equal(m.geom_rgba[dark], dark_before)
    assert np.array_equal(m.geom_rgba[m.geom_group == COLLISION_GROUP], collision_before)


def test_team_colours_match_the_control_ui():
    css = open("soccer_launcher/web/style.css").read()
    for name, rgba in (("--home", HOME), ("--away", AWAY)):
        hexes = [seg.split(":")[1].split(";")[0].strip() for seg in css.split(name)[1:2]]
        assert hexes and hexes[0] == "#%02x%02x%02x" % tuple(round(c * 255) for c in rgba[:3]), (name, hexes)
