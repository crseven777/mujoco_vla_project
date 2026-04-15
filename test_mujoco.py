import mujoco
import mujoco.viewer
import time

xml = """
<mujoco>
    <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="3 3 0.1"/>
        <body pos="0 0 1">
            <joint type="free"/>
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
    </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
