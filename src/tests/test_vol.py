from seg2mesh import vol


def test_create_canvas_for_volumes(simple_anatomies):
    vol.create_canvas_for_volumes(
        [simple_anatomies[s] for s in ("tibia", "tibia_cartilage", "femur", "femur_cartilage")], spacing=(0.5, 0.5, 0.5)
    )
