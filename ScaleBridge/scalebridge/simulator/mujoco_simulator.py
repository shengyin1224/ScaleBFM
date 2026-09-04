import os
import time
import torch
import imageio
import threading
import mujoco
import mujoco.viewer as mjv
import numpy as np
from loguru import logger
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from scalebridge.simulator.base_simulator import BaseSimulator
from scalebridge.utils.merge_robot_object_xml import merge_robot_object_xml

def draw_marker(pos,v,rgba=(1,0,0,1)):
    geom = v.user_scn.geoms[v.user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03,0.03,0.03],
        pos=pos,
        mat=np.eye(3).flatten(),
        rgba=list(rgba)
    )
    v.user_scn.ngeom += 1

class MujocoSimulator(BaseSimulator):
    def __init__(self, config, metadata_dict):
        
        self.record_video = config.get('record_video', False)
        self.marker = config.get('marker', False)
        self.use_joystick = config.get('joystick', False)
        self.camera_follow = config.get('camera_follow', False) or self.record_video
        self.reference_play_gate = config.get('reference_play_gate', False)
        self.reference_play_requested = False

        super().__init__(config, metadata_dict)

        self._setup_joystick()

    def _setup_backbone(self):
        super()._setup_backbone()

        xml_path = self.cfg.asset.xml_path

        # pase object from metadata if exists
        self.has_object = "object_names" in self.metadata_dict
        if self.has_object:
            xml_path = merge_robot_object_xml(xml_path, self.metadata_dict)

        self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep=self.low_dt
        
        self.viewer = mjv.launch_passive(
            model=self.mujoco_model,
            data=self.mujoco_data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self._key_callback,
        )
        self.marker_pos = None
        self.marker_rgba = None
        if self.record_video:
            save_dir = HydraConfig.get().runtime.output_dir
            video_name = os.path.join(save_dir, 'recording.mp4')
            self.video_writer = imageio.get_writer(video_name, fps=50)
            self.renderer = mujoco.Renderer(self.mujoco_model, height=480, width=640)
            self.render_scene = mujoco.MjvScene(self.mujoco_model, maxgeom=1000)


    def _setup_asset(self):
        super()._setup_asset()

        self.default_qpos = self.mujoco_data.qpos.copy()
        self.default_qvel = self.mujoco_data.qvel.copy()

        self.default_dof_pos = self.mujoco_data.qpos[7:7+self.num_joints].copy()

    def _setup_joystick(self):
        if self.use_joystick:
            import pygame
            logger.info(f'[Simulator] Using Joystick as command sender; Make sure you have connected the joystick to the PC!')
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                logger.info(f"[Simulator] Joystick detected: {self.joystick.get_name()}")
            else:
                pygame.quit()
                logger.error(f"[Simulator] Joystick undetected!")
            self.lin_vel_x_tmp = 0
            self.lin_vel_y_tmp = 0
            self.ang_vel_z_tmp = 0
            self.joystick_thread = threading.Thread(target=self._handle_joystick, args=(pygame,),daemon=True)
            self.joystick_thread.start()

    def _key_callback(self, keycode):
        if self.reference_play_gate and keycode == ord(' '):
            self.reference_play_requested = True

    def consume_reference_play_request(self):
        if self.reference_play_requested:
            self.reference_play_requested = False
            return True
        return False
    
    def _handle_joystick(self, pygame):
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
                self.lin_vel_x_tmp = self.joystick.get_axis(1) * -1
                self.lin_vel_y_tmp = self.joystick.get_axis(0) * -1
                self.ang_vel_z_tmp = self.joystick.get_axis(3) * -1
                time.sleep(0.001)
        except:
            pygame.quit()

    def _render(self):
        if not self.viewer:
            logger.info(f"[Simulator] Viewer has been closed; Automatically exit!")
            exit()

        if self.marker_pos is not None and self.marker:
            self.viewer.user_scn.ngeom = 0
            for i in range(self.marker_pos.shape[0]):
                draw_marker(
                    self.marker_pos[i], self.viewer, self._marker_rgba(i)
                )
            if self.record_video:
                self.render_scene.ngeom = 0
                for i in range(self.marker_pos.shape[0]):
                    geom = self.render_scene.geoms[self.render_scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=list(self._marker_rgba(i))
                    )
                    self.render_scene.ngeom += 1
        if self.camera_follow:
            self.viewer.cam.lookat = self.mujoco_data.qpos[:3].copy()
            self.viewer.cam.elevation = 0
            self.viewer.cam.azimuth = 180
            self.viewer.cam.distance = 3.0

        self.viewer.sync()

        if self.record_video:
            self.renderer.update_scene(self.mujoco_data, camera=self.viewer.cam)
        
            # Then manually add markers to the renderer's scene
            if self.marker_pos is not None and self.marker:
                # The renderer's scene might need to be updated after update_scene
                # So we add markers after the update
                for i in range(self.marker_pos.shape[0]):
                    geom = self.renderer.scene.geoms[self.renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.03, 0.03, 0.03],
                        pos=self.marker_pos[i],
                        mat=np.eye(3).flatten(),
                        rgba=list(self._marker_rgba(i))
                    )
                    self.renderer.scene.ngeom += 1

            img = self.renderer.render()
            self.video_writer.append_data(img)

    def _marker_rgba(self, index):
        """One colour per tracked body so the spheres can be told apart."""
        if self.marker_rgba is None or index >= len(self.marker_rgba):
            return (1, 0, 0, 1)
        return self.marker_rgba[index]

    def update_marker_pos(self, marker_pos, rgba=None):
        self.marker_pos = marker_pos.cpu().numpy()
        if rgba is not None:
            self.marker_rgba = tuple(tuple(colour) for colour in rgba)

    def refresh_sim(self):

        state_dict = { # weishuai: FIXME
            "root_pos": self.mujoco_data.qpos[:3],
            "root_quat_wxyz": self.mujoco_data.qpos[3:7],
            "base_ang_vel": self.mujoco_data.qvel[3:6],
            "dof_pos": self.mujoco_data.qpos[7:7+self.num_joints][self.sim_to_env_joint_idx],
            "dof_vel": self.mujoco_data.qvel[6:6+self.num_joints][self.sim_to_env_joint_idx],
        }

        if self.has_object:
            object_root_pose = self.mujoco_data.qpos[7+self.num_joints:].reshape(-1, 7)
            state_dict.update({
                "object_root_pos": object_root_pose[:, :3],
                "object_root_quat_wxyz": object_root_pose[:, 3:7],
            })

        if self.use_joystick:
            self.commands = np.array([self.lin_vel_x_tmp, self.lin_vel_y_tmp, self.ang_vel_z_tmp], dtype=np.float32)
            self.commands = np.where(np.abs(self.commands) < 0.05, 0, self.commands)
            state_dict.update({'commands': self.commands})
        
        return {k:torch.from_numpy(v).float() for k,v in state_dict.items()}

    def calibrate(self, init_state_dict = {}):
        init_qpos = self.default_qpos.copy()
        init_qvel = self.default_qvel.copy()

        if "root_pos" in init_state_dict:
            init_qpos[:3] = init_state_dict["root_pos"]
        if "root_quat" in init_state_dict:
            init_qpos[3:7] = init_state_dict["root_quat"]
        if "dof_pos" in init_state_dict:
            init_qpos[7:7+self.num_joints][self.env_action_to_sim_idx] = init_state_dict["dof_pos"]
        if "root_lin_vel" in init_state_dict:
            init_qvel[:3] = init_state_dict["root_lin_vel"]
        if "root_ang_vel" in init_state_dict:
            init_qvel[3:6] = init_state_dict["root_ang_vel"]
        if "dof_vel" in init_state_dict:
            init_qvel[6:6+self.num_joints][self.env_action_to_sim_idx] = init_state_dict["dof_vel"]
        
        if self.has_object:
            
            init_obj_pose = init_qpos[7+self.num_joints:].copy().reshape(-1, 7)
            init_obj_vel = init_qvel[6+self.num_joints:].copy().reshape(-1, 6)
            
            if "object_root_pos" in init_state_dict:
                init_obj_pose[:, :3] = init_state_dict["object_root_pos"]
            if "object_root_quat" in init_state_dict:
                init_obj_pose[:, 3:7] = init_state_dict["object_root_quat"]
            if "object_root_lin_vel" in init_state_dict:
                init_obj_vel[:, :3] = init_state_dict["object_root_lin_vel"]
            if "object_root_ang_vel" in init_state_dict:
                init_obj_vel[:, 3:] = init_state_dict["object_root_ang_vel"]

            init_qpos[7+self.num_joints:] = init_obj_pose.flatten()
            init_qvel[6+self.num_joints:] = init_obj_vel.flatten()

        self.mujoco_data.qpos[:] = init_qpos
        self.mujoco_data.qvel[:] = init_qvel
        self.mujoco_data.ctrl[:] = 0

        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)

        return init_qpos[:3], init_qpos[3:7]

    def apply_action(self, tgt_dof_pos):
        tgt_dof_pos = tgt_dof_pos.squeeze()

        target_dof_pos_in_sim = self.default_dof_pos.copy()
        target_dof_pos_in_sim[self.env_action_to_sim_idx] = tgt_dof_pos

        for _ in range(self.decimation):
            torque = (target_dof_pos_in_sim - self.mujoco_data.qpos[7:7+self.num_joints]) * self.stiffness - self.mujoco_data.qvel[6:6+self.num_joints] * self.damping # no clip here
            # torque = np.clip(torque, -self.torque_limit, self.torque_limit)
            self.mujoco_data.ctrl[:] = torque # weishuai: no clip applied here
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

        self._render()

    def _get_joint_names(self):
        joint_names = []
        for i in range(self.mujoco_model.nu):
            dof_name = mujoco.mj_id2name(self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_names.append(dof_name)
        return joint_names
