# Recreate the static routing workcell in another repository

This note is a portable snapshot of the setup on 2026-10-20. It contains a
complete MuJoCo XML and all camera numbers needed to reproduce the layout
without importing this repository, `curobo_example/`, Menagerie, the calibration
ZIP, meshes or robot assets.

The included scene has the **two adjacent tables**, camera tower and camera
housings, the calibrated top/side cameras, and fixed robot base frames. There
are no robot links, grippers, routing fixtures, crimp-machine blocks, DLO,
actuators, constraints or task code. The floor and overview camera are retained
for inspection. Base-frame axes are non-colliding sites, not robot geometry.
The small robot mounting plates are omitted; their height is already reflected
in the base poses.

## 1. Coordinate and dimension conventions

- Distances are metres, angles radians, and quaternions are **w x y z**.
- World +Z points upward. The floor is Z = 0.
- A transform `world_from_frame` maps column-vector coordinates from that frame
  into world coordinates: `p_world = R @ p_frame + t`.
- MuJoCo box `size` values are **half extents**. Other engines usually want
  full dimensions, so double all three values for a box.
- Physics uses a 0.0005 s step, gravity (0, 0, -9.81), and `implicitfast`.
  This static export has no controls. If matching the original control cadence
  later, four physics steps make one 0.002 s control interval.

| Component | World centre/origin (m) | Full dimensions (m) |
| --- | --- | --- |
| Table A frame | (0, -0.5, 0) | Top: 1.5 × 1.0 × 0.1 |
| Table B frame | (0, +0.5, 0) | Top: 1.5 × 1.0 × 0.1 |
| Table A top centre | (0, -0.5, 0.78597) | 1.5 × 1.0 × 0.1 |
| Table B top centre | (0, +0.5, 0.78597) | 1.5 × 1.0 × 0.1 |
| Each of eight legs | See XML, relative to its table | 0.07 × 0.07 × 0.73597 |
| Camera tower post | (0.675, -0.675, 1.33597) | 0.2 × 0.2 × 1.0; rotated |
| Camera tower beam | (0.514538749, -0.514538749, 1.88597) | 0.2 × 0.2 × 1.0; rotated |

The combined tabletop covers X = [-0.75, 0.75], Y = [-1.0, 1.0], with a seam
at Y = 0 and its upper surface at **Z = 0.83597 m**. Keep the two separate
solids to reproduce the original collision layout. The tower geometry and its
rotations below are the retained CAD collision proxies.

## 2. Copy this XML into the target repository

Save the following block as `workcell_static.xml`. It is a complete standalone
file. Both camera body poses are already **world-from-optical** transforms;
the child camera rotation converts the optical axes to MuJoCo camera axes.
Do not add an extra frame conversion on top of the one in this XML.

```xml
<mujoco model="routing_workcell_static_setup">
  <compiler angle="radian" autolimits="true" />
  <option timestep="0.0005" integrator="implicitfast" gravity="0 0 -9.81" />
  <visual>
    <global offwidth="1280" offheight="900" />
    <headlight diffuse="0.7 0.7 0.7" ambient="0.3 0.3 0.3" />
    <map znear="0.01" />
  </visual>
  <default>
    <geom friction="0.8 0.005 0.0001" solref="0.01 1" solimp="0.95 0.99 0.001" />
  </default>
  <worldbody>
    <light pos="0 -1 4" dir="0 0 -1" diffuse="0.8 0.8 0.8" />
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.22 0.25 0.28 1" />
    <camera name="overview" pos="3 -4 3.5" xyaxes="0.8 0.6 0 -0.2462 0.3283 0.9119" />
    <body name="table_A" pos="0 -0.5 0">
      <geom name="table_A_top" type="box" pos="0 0 0.78597" size=".75 .5 .05" rgba="0.13 0.15 0.17 1" />
      <geom name="table_A_leg_0" type="box" pos="-0.7 -0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
      <geom name="table_A_leg_1" type="box" pos="-0.7 0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
      <geom name="table_A_leg_2" type="box" pos="0.7 -0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
      <geom name="table_A_leg_3" type="box" pos="0.7 0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
    </body>
    <body name="table_B" pos="0 0.5 0">
      <geom name="table_B_top" type="box" pos="0 0 0.78597" size=".75 .5 .05" rgba="0.13 0.15 0.17 1" />
      <geom name="table_B_leg_0" type="box" pos="-0.7 -0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
      <geom name="table_B_leg_1" type="box" pos="-0.7 0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
      <geom name="table_B_leg_2" type="box" pos="0.7 -0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
      <geom name="table_B_leg_3" type="box" pos="0.7 0.45 0.367985" size=".035 .035 .367985" rgba="0.13 0.15 0.17 1" />
    </body>
    <body name="camera_tower_post" pos="0.675 -0.675 1.33597" quat="0.923879526 0 0 -0.382683448">
      <geom name="camera_tower_post" type="box" pos="0 0 0" size="0.100000001 0.100000001 0.5" rgba="0.58 0.62 0.66 1" />
    </body>
    <body name="camera_tower_beam" pos="0.514538749 -0.514538749 1.88597" quat="-0.653281482 0.27059805 0.653281482 0.27059805">
      <geom name="camera_tower_beam" type="box" pos="0 0 0" size="0.100000001 0.100000001 0.5" rgba="0.58 0.62 0.66 1" />
    </body>
    <body name="side_camera" pos="0.564211348501 -0.61049739804 1.31888152545" quat="0.572040879577 -0.732168390444 -0.297763687312 0.219169949229">
      <geom name="side_camera" type="box" pos="0 0 -.027" size=".045 .0175 .015" rgba=".16 .18 .21 1" />
      <geom name="side_capture_lens" type="cylinder" pos="0 0 -.005" size=".012 .005" rgba=".1 .45 .7 1" contype="0" conaffinity="0" group="2" />
      <camera name="side_capture" pos="0 0 0" quat="0 1 0 0" resolution="1280 720" sensorsize="1.77777777778 1" focalpixel="920.841389973 921.098" principalpixel="-14.4433533333 5.200114" ipd="0" />
      <site name="side_capture_optical" type="sphere" size=".003" rgba="1 .6 .1 1" group="4" />
    </body>
    <body name="overhead_camera" pos="0.125556233775 -0.1374434363 1.98659116858" quat="-0.139457649191 0.924171738836 0.350470542142 -0.0602375324086">
      <geom name="overhead_camera" type="box" pos="0 0 -.027" size=".045 .0175 .015" rgba=".16 .18 .21 1" />
      <geom name="top_lens" type="cylinder" pos="0 0 -.005" size=".012 .005" rgba=".1 .45 .7 1" contype="0" conaffinity="0" group="2" />
      <camera name="top" pos="0 0 0" quat="0 1 0 0" resolution="1280 720" sensorsize="1.77777777778 1" focalpixel="916.59000651 915.54280599" principalpixel="-7.69689941406 4.4335530599" ipd="0" />
      <site name="top_optical" type="sphere" size=".003" rgba="1 .6 .1 1" group="4" />
    </body>
    <site name="zed_optical" type="sphere" size=".006" rgba=".3 .8 .4 1" group="4" pos="0.536070970271 -0.631697081668 1.46688544678" quat="0.572919735609 -0.715003257978 -0.332398258133 0.223706762572" />
    <site name="multiview_reference_optical" type="sphere" size=".006" rgba=".3 .8 .4 1" group="4" pos="0.561356904794 -0.608732363789 1.31878983171" quat="0.57369524319 -0.735462782298 -0.292260615836 0.211073437994" />
    <body name="robotA_base_link" pos="-0.425000012 -0.574999988 0.85097003" quat="-0.707106733 0 0 0.707106829">
      <site name="robotA_base_link_x" type="capsule" fromto="0 0 0 0.08 0 0" size="0.002" rgba="1 0.15 0.15 1" />
      <site name="robotA_base_link_y" type="capsule" fromto="0 0 0 0 0.08 0" size="0.002" rgba="0.15 1 0.15 1" />
      <site name="robotA_base_link_z" type="capsule" fromto="0 0 0 0 0 0.08" size="0.002" rgba="0.15 0.4 1 1" />
    </body>
    <body name="robotB_base_link" pos="0.424999982 0.574999928 0.85097003" quat="-0.707106733 0 0 0.707106829">
      <site name="robotB_base_link_x" type="capsule" fromto="0 0 0 0.08 0 0" size="0.002" rgba="1 0.15 0.15 1" />
      <site name="robotB_base_link_y" type="capsule" fromto="0 0 0 0 0.08 0" size="0.002" rgba="0.15 1 0.15 1" />
      <site name="robotB_base_link_z" type="capsule" fromto="0 0 0 0 0 0.08" size="0.002" rgba="0.15 0.4 1 1" />
    </body>
    <body name="calibration_robot_b" pos="0.425 0.575 0.85097" quat="0.7071067811865476 0 0 0.7071067811865476" />
  </worldbody>
</mujoco>
```

The camera housings are schematic 90 × 35 × 30 mm boxes, offset behind their
optical origins; they are not measured camera CAD. Tower/housing geometry can
be omitted if only the rendered views and coordinate frames are needed. Their
removal does not change the camera poses.

## 3. Robot base frames: preserve the frame convention

`robotA_base_link` and `robotB_base_link` reproduce the **actual base-link
frames in our MuJoCo model**, including the original small float-rounding
residuals. The red, green and blue sites show local +X, +Y and +Z respectively.
Their axes are 80 mm long and have no collision geometry.

| Frame | World position (m), rounded | World yaw, rounded |
| --- | --- | --- |
| `robotA_base_link` | (-0.425, -0.575, 0.85097) | -90° |
| `robotB_base_link` | (+0.425, +0.575, 0.85097) | -90° |
| `calibration_robot_b` | (+0.425, +0.575, 0.85097) | +90° |

Use the **full-precision `pos` and `quat` in the XML** for exact matching.
The nominal base height is 15 mm above the tabletop. If mounting plates are
wanted later, each is a 0.25 × 0.25 × 0.015 m box, centred at approximately
(-0.425, -0.575, 0.84347) and (+0.425, +0.575, 0.84347) m.

The `calibration_robot_b` frame is different from `robotB_base_link`.
The retained camera calibration is registered to the authored USD robot-B
**parent/mount frame**, whose yaw is +90°. Its nominal transform is:

```text
world_from_calibration_robot_b =
[[ 0, -1,  0, 0.425   ],
 [ 1,  0,  0, 0.575   ],
 [ 0,  0,  1, 0.85097 ],
 [ 0,  0,  0, 1       ]]
```

Consequently, relative camera extrinsics from the old calibration must be
composed with this +90° frame, **not** with the -90° base-link frame. The
camera poses in this note are already composed into world coordinates.
The camera data also contained an independently fitted robot-A pose; we did
not move the CAD robot mounts to that fit. Keep the exported CAD base frames
when reproducing this scene.

For a future UR model attachment, check the imported model's root rotation:
a model with its own 180° root rotation needs a corresponding mount-frame
conversion. Do not apply both that rotation and the exported base-link
rotation without accounting for their different frame definitions.

## 4. Camera calibration for other renderers

Optical coordinates are **+X right, +Y down, +Z forward**. The matrices below
are `world_from_optical`, with translation in the last column. They can be
used directly with an OpenCV-style camera convention.

### `top` optical pose

```text
[ 0.747083677561,  0.630988771358, -0.209091245974,  0.125556233775]
[ 0.664591110009, -0.715443926346,  0.215542675013, -0.137443436300]
[-0.013588054303, -0.299988597575, -0.953846007543,  1.986591168575]
[ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]
```

### `side_capture` optical pose

```text
[ 0.726602639747,  0.185277978276, -0.661604621114,  0.564211348501]
[ 0.686774660411, -0.168212037222,  0.707138795676, -0.610497398040]
[ 0.019727385270, -0.968182204598, -0.249467530895,  1.318881525455]
[ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]
```

### `zed` optical pose

```text
[ 0.678933364740,  0.218999636488, -0.700777172475,  0.536070970271]
[ 0.731663713556, -0.122548749079,  0.670559478618, -0.631697081668]
[ 0.060972916191, -0.967998431464, -0.243436521857,  1.466885446779]
[ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]
```

### `multiview_reference` optical pose

```text
[ 0.740063472409,  0.187709956676, -0.645810366106,  0.561356904794]
[ 0.672077266040, -0.170915000745,  0.720486093546, -0.608732363789]
[ 0.024863734201, -0.967239905445, -0.252643543430,  1.318789831712]
[ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000]
```

Only `top` and `side_capture` have matching intrinsics and are rendered
calibrated cameras. `zed_optical` and `multiview_reference_optical` are pose
markers only; they represent distinct calibrations and should not inherit
`side_capture` intrinsics.

At the scene's 1280 × 720 stream resolution:

| Camera | fx (px) | fy (px) | cx (px) | cy (px) |
| --- | ---: | ---: | ---: | ---: |
| `top` | 916.590006510417 | 915.542805989583 | 647.696899414062 | 355.566446940104 |
| `side_capture` | 920.841389973333 | 921.098000000000 | 654.443353333333 | 354.799886000000 |

The original top calibration was at 1920 × 1080, with
(fx, fy, cx, cy) = (1374.885009765625, 1373.314208984375,
971.5453491210938, 533.3496704101562). Its stream-resolution values are scaled
by 2/3. The side calibration was already at 1280 × 720.

For another resolution without cropping, scale fx and cx by new_width/old_width,
and fy and cy by new_height/old_height. There are no distortion coefficients;
the simulation uses an ideal pinhole model. The third camera, `overview`, is
an uncalibrated inspection camera and does not represent a physical sensor.

For an optical point (X, Y, Z), projection is
`u = fx * X/Z + cx`, `v = fy * Y/Z + cy`. MuJoCo's camera looks along its
local **-Z**, with local +Y upward, so its world pose is:

```text
world_from_mujoco_camera = world_from_optical @ diag(1, -1, -1, 1)
```

The XML implements this with the child camera's `quat="0 1 0 0"`.
`principalpixel` is **not** the OpenCV principal point (cx, cy): for these
MuJoCo camera definitions use `(width/2 - cx, height/2 - cy)`. The XML has
those offsets already. Use the intrinsic camera attributes from the XML
rather than replacing them with an approximate field of view.

## 5. Load and verify the exported scene

Use a Python environment with `mujoco` and `numpy` installed. This export was
compiled and checked with MuJoCo **3.11.0** and Python **3.11**. No project
modules, GPU packages or robot assets are required. Keep package installation
within the target repository's own dependency workflow.

Save this as `check_workcell.py` next to `workcell_static.xml` and run
`python check_workcell.py`:

```python
from pathlib import Path

import mujoco
import numpy as np

path = Path(__file__).with_name("workcell_static.xml")
model = mujoco.MjModel.from_xml_path(str(path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

assert model.nq == model.nv == model.nu == 0
assert model.neq == 0 and model.nmesh == 0
assert model.ncam == 3
assert model.opt.timestep == 0.0005
for name in ("table_A_top", "table_B_top"):
    geom = model.geom(name)
    np.testing.assert_allclose(data.geom_xpos[geom.id, 2] + geom.size[2], 0.83597)
for name, position in (
    ("robotA_base_link", [-0.425000012, -0.574999988, 0.85097003]),
    ("robotB_base_link", [0.424999982, 0.574999928, 0.85097003]),
):
    np.testing.assert_allclose(data.body(name).xpos, position, atol=1e-12)
for name, optical in (("top", "top_optical"), ("side_capture", "side_capture_optical")):
    camera = model.camera(name).id
    site = model.site(optical).id
    np.testing.assert_allclose(data.cam_xpos[camera], data.site_xpos[site], atol=1e-12)
    optical_rotation = data.site_xmat[site].reshape(3, 3)
    camera_rotation = data.cam_xmat[camera].reshape(3, 3)
    np.testing.assert_allclose(
        camera_rotation, optical_rotation @ np.diag([1, -1, -1]), atol=1e-12
    )
print(f"Static workcell verified: {model.nbody} bodies, {model.ngeom} geoms")
```

For interactive inspection, save and run this separate `view_workcell.py`:

```python
import time
from pathlib import Path

import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path(
    str(Path(__file__).with_name("workcell_static.xml"))
)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0, 0, 1.1]
    viewer.cam.distance = 4.2
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -25
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)  # This setup is static; stepping physics is unnecessary.
```

To render calibrated images, select `top` or `side_capture` by name and render
at 1280 × 720. A pinhole-projection consistency check is preferable to judging
calibration from the overview camera or the schematic housing shape.

## Source and scope

Values were extracted from `envs/routing_workcell/scene_pegs_only.xml` and
`envs/routing_workcell/camera_calibration.json`; these components are shared
with the mixed workcell variant. The calibration originated in the user-supplied
`scene_setup_reconstruction.zip`, registered to the authored workcell world.
This note embeds the necessary values so none of those files are needed in
the target repository. Camera registration retains the original CAD-to-robot-B
alignment; it is not a new surveyed world calibration.
