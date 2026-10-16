"""How robots look in the MuJoCo viewer of a match: cheap team-coloured bodies instead of ~1 M-triangle meshes.

One T2's visual meshes are ~990,000 triangles; ten of them (plus shadows) take the viewer over a second a frame
in software GL and leave it trailing the sim. The collision shapes each robot already has (boxes, cylinders) draw in
milliseconds, so by default the viewer shows those, tinted by team. Pure MuJoCo, no ROS.
"""
import numpy as np

# Home blue and away orange: the same colours as --home / --away in soccer_launcher/web/style.css (dark theme).
TEAM_RGBA = {"home": (77 / 255, 163 / 255, 1.0, 1.0), "away": (1.0, 159 / 255, 67 / 255, 1.0)}  # #4da3ff, #ff9f43
VISUAL_GROUP, COLLISION_GROUP = 1, 3  # booster_assets T2 MJCF: meshes in group 1, collision shapes in group 3
LIGHT_MESH_MIN_RED = 0.5  # the T2 meshes are light grey (0.85) with a few dark parts (0.21): tint only the light ones


def apply_look(model, tints: dict[int, tuple], detailed: bool = False) -> dict[int, int]:
    """Tint each robot by team and switch shadows off. Returns the geom-group visibility the viewer should use.

    tints maps a robot's root body id (model.body_rootid of its trunk) to an RGBA. detailed=False shows the cheap
    collision shapes; detailed=True keeps the real meshes and tints only their light-grey parts.
    """
    model.light_castshadow[:] = 0  # shadows are about half of what is left to draw
    root = model.body_rootid[model.geom_bodyid]
    for root_id, rgba in tints.items():
        if detailed:
            sel = (root == root_id) & (model.geom_group == VISUAL_GROUP) & (model.geom_rgba[:, 0] > LIGHT_MESH_MIN_RED)
        else:
            sel = (root == root_id) & (model.geom_group == COLLISION_GROUP)
        model.geom_rgba[sel] = np.asarray(rgba, dtype=np.float32)
        model.geom_matid[sel] = -1
    return {VISUAL_GROUP: 1, COLLISION_GROUP: 0} if detailed else {VISUAL_GROUP: 0, COLLISION_GROUP: 1}
