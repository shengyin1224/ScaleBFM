import atexit
import json
import os
import time

import imageio
import mujoco
import mujoco.viewer as mjv
import numpy as np
from hydra.core.hydra_config import HydraConfig
from loguru import logger

from scalebridge.simulator.mujoco_simulator import MujocoSimulator
from scalebridge.utils.hand_ik_assets import DEFAULT_SEARCH_DIR, resolve_hand_ik_path


INSPIRE_DOF_NAMES = [
    "index_proximal_joint", "index_intermediate_joint",
    "middle_proximal_joint", "middle_intermediate_joint",
    "pinky_proximal_joint", "pinky_intermediate_joint",
    "ring_proximal_joint", "ring_intermediate_joint",
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint", "thumb_distal_joint",
]


def interpolate(values, frame):
    frame = float(np.clip(frame, 0.0, len(values) - 1))
    lower = int(np.floor(frame))
    upper = min(lower + 1, len(values) - 1)
    alpha = frame - lower
    return values[lower] * (1.0 - alpha) + values[upper] * alpha


class InspireMujocoSimulator(MujocoSimulator):
    """ScaleBridge body control plus kinematic Inspire IK playback."""

    def _setup_backbone(self):
        self.low_dt = self.cfg.low_dt
        self.decimation = self.cfg.decimation
        self.high_dt = self.low_dt * self.decimation
        logger.info(f"[Simulator] Robot-level Control Frequency Set to {1/self.low_dt}HZ")
        logger.info(f"[Simulator] Policy-level Control Frequency Set to {1/self.high_dt}HZ")

        xml_path = self.cfg.inspire_xml_path
        self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        self._align_with_scalebridge()
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep = self.low_dt
        self.viewer = mjv.launch_passive(
            model=self.mujoco_model,
            data=self.mujoco_data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self._key_callback,
        )
        self.marker_pos = None
        self.marker_rgba = None
        self.record_online_trace = bool(self.cfg.get("record_online_trace", False))
        self._online_trace_streams = {}
        self._online_trace_last_flush = 0.0
        self.last_body_torque = np.zeros(
            len(self.metadata_dict["joint_names"]), dtype=np.float64
        )
        if self.record_online_trace:
            atexit.register(self._close_online_trace)
            logger.info(
                "[Simulator] Online HAT state/chunk/policy trace recording enabled."
            )
        if self.record_video:
            save_dir = HydraConfig.get().runtime.output_dir
            self.video_writer = imageio.get_writer(
                os.path.join(save_dir, "recording.mp4"), fps=50
            )
            self.renderer = mujoco.Renderer(self.mujoco_model, height=480, width=640)
            self.render_scene = mujoco.MjvScene(self.mujoco_model, maxgeom=1000)

    def _align_with_scalebridge(self):
        """Keep the stock ScaleBridge body model; only Inspire hands may differ."""
        reference = mujoco.MjModel.from_xml_path(self.cfg.asset.xml_path)
        model = self.mujoco_model
        self._copy_body_dynamics(reference, model)

        model.vis.headlight.ambient[:] = reference.vis.headlight.ambient
        model.vis.headlight.diffuse[:] = reference.vis.headlight.diffuse
        model.vis.headlight.specular[:] = reference.vis.headlight.specular
        model.vis.rgba.haze[:] = reference.vis.rgba.haze
        model.vis.global_.azimuth = reference.vis.global_.azimuth
        model.vis.global_.elevation = reference.vis.global_.elevation
        model.vis.global_.offwidth = reference.vis.global_.offwidth
        model.vis.global_.offheight = reference.vis.global_.offheight
        model.stat.meansize = reference.stat.meansize
        if model.nlight:
            model.light_active[:] = False

        source_texture = mujoco.mj_name2id(
            reference, mujoco.mjtObj.mjOBJ_TEXTURE, "groundplane"
        )
        target_texture = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_TEXTURE, "groundplane"
        )
        source_size = int(
            reference.tex_width[source_texture]
            * reference.tex_height[source_texture]
            * reference.tex_nchannel[source_texture]
        )
        target_size = int(
            model.tex_width[target_texture]
            * model.tex_height[target_texture]
            * model.tex_nchannel[target_texture]
        )
        if source_size != target_size:
            raise ValueError("ScaleBridge and Inspire ground textures are incompatible")
        source_start = int(reference.tex_adr[source_texture])
        target_start = int(model.tex_adr[target_texture])
        model.tex_data[target_start:target_start + target_size] = reference.tex_data[
            source_start:source_start + source_size
        ]
        sky_size = int(model.tex_width[0] * model.tex_height[0] * model.tex_nchannel[0])
        sky_start = int(model.tex_adr[0])
        model.tex_data[sky_start:sky_start + sky_size] = 255

        source_material = mujoco.mj_name2id(
            reference, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane"
        )
        target_material = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane"
        )
        model.mat_rgba[target_material] = reference.mat_rgba[source_material]
        model.mat_reflectance[target_material] = reference.mat_reflectance[source_material]
        source_floor = mujoco.mj_name2id(
            reference, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        target_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        model.geom_size[target_floor] = reference.geom_size[source_floor]

    @staticmethod
    def _copy_body_dynamics(reference, model):
        """Copy all common 29-DOF body dynamics from the stock ScaleBridge G1.

        The external Inspire asset has useful hand geometry, but its body joint
        damping, friction, actuator limits, and wrist inertias differ from the
        policy's stock simulation model.  Match common named bodies, joints and
        actuators while leaving the additional Inspire hand elements untouched.
        """
        common_bodies = 0
        for source_id in range(reference.nbody):
            name = mujoco.mj_id2name(reference, mujoco.mjtObj.mjOBJ_BODY, source_id)
            if name is None:
                continue
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if target_id < 0:
                continue
            for attr in (
                "body_mass", "body_inertia", "body_ipos", "body_iquat",
                "body_pos", "body_quat", "body_gravcomp",
            ):
                getattr(model, attr)[target_id] = getattr(reference, attr)[source_id]
            common_bodies += 1

        common_joints = 0
        for source_id in range(reference.njnt):
            name = mujoco.mj_id2name(reference, mujoco.mjtObj.mjOBJ_JOINT, source_id)
            if name is None:
                continue
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if target_id < 0:
                continue
            for attr in (
                "jnt_pos", "jnt_axis", "jnt_range", "jnt_stiffness",
                "jnt_margin", "jnt_solref", "jnt_solimp", "jnt_limited",
            ):
                getattr(model, attr)[target_id] = getattr(reference, attr)[source_id]
            source_dof = int(reference.jnt_dofadr[source_id])
            target_dof = int(model.jnt_dofadr[target_id])
            for attr in (
                "dof_damping", "dof_frictionloss", "dof_armature",
                "dof_solref", "dof_solimp",
            ):
                getattr(model, attr)[target_dof] = getattr(reference, attr)[source_dof]
            common_joints += 1

        common_actuators = 0
        for source_id in range(reference.nu):
            name = mujoco.mj_id2name(
                reference, mujoco.mjtObj.mjOBJ_ACTUATOR, source_id
            )
            if name is None:
                continue
            target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if target_id < 0:
                continue
            for attr in (
                "actuator_ctrlrange", "actuator_forcerange", "actuator_gear",
                "actuator_dynprm", "actuator_gainprm", "actuator_biasprm",
                "actuator_lengthrange", "actuator_ctrllimited",
                "actuator_forcelimited",
            ):
                getattr(model, attr)[target_id] = getattr(reference, attr)[source_id]
            common_actuators += 1

        if common_joints != 29 or common_actuators != 29:
            raise RuntimeError(
                "Failed to align the complete ScaleBridge body: "
                f"bodies={common_bodies}, joints={common_joints}, "
                f"actuators={common_actuators}"
            )
        logger.info(
            "[Simulator] Matched stock ScaleBridge dynamics for 29 body joints; "
            "only the Inspire hand bodies and joints remain asset-specific."
        )

    def _setup_asset(self):
        self.body_joint_names = list(self.metadata_dict["joint_names"])
        self.num_joints = len(self.body_joint_names)
        self.body_qpos_adr = self._joint_addresses(
            self.body_joint_names, self.mujoco_model.jnt_qposadr
        )
        self.body_dof_adr = self._joint_addresses(
            self.body_joint_names, self.mujoco_model.jnt_dofadr
        )
        self.body_actuator_ids = self._actuator_ids(self.body_joint_names)
        self.stiffness = np.asarray(self.metadata_dict["stiffness"], dtype=np.float64)
        self.damping = np.asarray(self.metadata_dict["damping"], dtype=np.float64)
        self.default_qpos = self.mujoco_data.qpos.copy()
        self.default_qvel = self.mujoco_data.qvel.copy()
        stock_model = mujoco.MjModel.from_xml_path(self.cfg.asset.xml_path)
        stock_data = mujoco.MjData(stock_model)
        # The Inspire asset ships with a rotated capture pose.  Copy both the
        # root and all 29 named body-joint defaults from the stock simulator.
        self.default_qpos[:7] = stock_data.qpos[:7]
        for name, target_address in zip(self.body_joint_names, self.body_qpos_adr):
            source_joint = mujoco.mj_name2id(
                stock_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            source_address = int(stock_model.jnt_qposadr[source_joint])
            self.default_qpos[target_address] = stock_data.qpos[source_address]
        self.mujoco_data.qpos[:] = self.default_qpos
        self.default_dof_pos = self.mujoco_data.qpos[self.body_qpos_adr].copy()

        self.left_hand_names = [f"L_{name}" for name in INSPIRE_DOF_NAMES]
        self.right_hand_names = [f"R_{name}" for name in INSPIRE_DOF_NAMES]
        self.hand_joint_names = self.left_hand_names + self.right_hand_names
        self.hand_qpos_adr = self._joint_addresses(
            self.hand_joint_names, self.mujoco_model.jnt_qposadr
        )
        self.hand_dof_adr = self._joint_addresses(
            self.hand_joint_names, self.mujoco_model.jnt_dofadr
        )

        self.hand_online_targets = bool(self.cfg.get("hand_online_targets", False))
        if self.hand_online_targets:
            self.hand_fps = 50.0
            self.left_hand_reference = np.zeros((1, 12), dtype=np.float64)
            self.right_hand_reference = np.zeros((1, 12), dtype=np.float64)
            self.left_hand_target = self.left_hand_reference[0].copy()
            self.right_hand_target = self.right_hand_reference[0].copy()
            logger.info("[Simulator] Inspire hands accept online fingertip IK targets.")
        else:
            hand_path = resolve_hand_ik_path(
                self.cfg.hand_ik_path,
                self.cfg.motion_path,
                self.cfg.get("hand_ik_search_dir", DEFAULT_SEARCH_DIR),
            )
            if not hand_path:
                raise ValueError("Inspire Sim2Sim requires simulator.config.hand_ik_path")
            hand = np.load(hand_path, allow_pickle=False)
            for key in ("fps", "lh_dof_pos", "rh_dof_pos"):
                if key not in hand:
                    raise KeyError(f"Hand IK file is missing {key}: {hand_path}")
            self.hand_fps = float(hand["fps"])
            self.left_hand_reference = np.asarray(hand["lh_dof_pos"], dtype=np.float64)
            self.right_hand_reference = np.asarray(hand["rh_dof_pos"], dtype=np.float64)
            logger.info(f"[Simulator] Loaded Inspire hand IK from {hand_path}")
        self.hand_elapsed = 0.0
        self._set_hand_pose(0.0)

    def _joint_addresses(self, names, address_array):
        result = []
        for name in names:
            joint_id = mujoco.mj_name2id(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise KeyError(f"Joint not found in Inspire XML: {name}")
            result.append(int(address_array[joint_id]))
        return np.asarray(result, dtype=np.int64)

    def _actuator_ids(self, names):
        result = []
        for name in names:
            actuator_id = mujoco.mj_name2id(
                self.mujoco_model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if actuator_id < 0:
                raise KeyError(f"Actuator not found in Inspire XML: {name}")
            result.append(actuator_id)
        return np.asarray(result, dtype=np.int64)

    def _set_hand_pose(self, elapsed):
        if self.hand_online_targets:
            target = np.concatenate((self.left_hand_target, self.right_hand_target))
        else:
            frame = elapsed * self.hand_fps
            target = np.concatenate((
                interpolate(self.left_hand_reference, frame),
                interpolate(self.right_hand_reference, frame),
            ))
        self.mujoco_data.qpos[self.hand_qpos_adr] = target
        self.mujoco_data.qvel[self.hand_dof_adr] = 0.0

    def set_hand_ik_target(self, left_dof_pos, right_dof_pos):
        if not self.hand_online_targets:
            raise RuntimeError("Simulator is configured for offline hand playback")
        left = np.asarray(left_dof_pos, dtype=np.float64)
        right = np.asarray(right_dof_pos, dtype=np.float64)
        if left.shape != (12,) or right.shape != (12,):
            raise ValueError(f"Online Inspire targets must be (12,), got {left.shape}/{right.shape}")
        self.left_hand_target = left.copy()
        self.right_hand_target = right.copy()

    @staticmethod
    def _jsonable(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: InspireMujocoSimulator._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [InspireMujocoSimulator._jsonable(item) for item in value]
        raise TypeError(f"Cannot serialize online trace value of type {type(value).__name__}")

    def _online_trace_write(self, filename, record):
        if not self.record_online_trace:
            return
        stream = self._online_trace_streams.get(filename)
        if stream is None:
            save_dir = HydraConfig.get().runtime.output_dir
            path = os.path.join(save_dir, filename)
            stream = open(path, "w", encoding="utf-8")
            self._online_trace_streams[filename] = stream
            logger.info(f"[Simulator] Recording online trace to {path}")
        stream.write(json.dumps(self._jsonable(record), separators=(",", ":")) + "\n")
        now = time.monotonic()
        if now - self._online_trace_last_flush >= 1.0:
            for active in self._online_trace_streams.values():
                active.flush()
            self._online_trace_last_flush = now

    def _close_online_trace(self):
        streams, self._online_trace_streams = self._online_trace_streams, {}
        for stream in streams.values():
            try:
                stream.flush()
                stream.close()
            except OSError:
                pass

    def record_hat_robot_state(self, state):
        self._online_trace_write("online_hat_states.jsonl", state)

    def record_hat_chunk(self, raw_chunk, aligned_chunk):
        self._online_trace_write(
            "online_hat_chunks.jsonl",
            {
                "record_type": "hat_chunk",
                "time": time.time(),
                "monotonic": time.monotonic(),
                "raw": raw_chunk,
                "aligned": aligned_chunk,
            },
        )

    def record_hat_online_step(
        self, *, requested_target_dof_pos, applied_target_dof_pos,
        policy_action, reference_sample, policy_observation, chunk_sequence_id,
        reference_playing, takeover_alpha, localization_paused,
    ):
        self._online_trace_write(
            "online_policy_trace.jsonl",
            {
                "record_type": "policy_step",
                "time": time.time(),
                "monotonic": time.monotonic(),
                "chunk_sequence_id": chunk_sequence_id,
                "reference_playing": reference_playing,
                "takeover_alpha": takeover_alpha,
                "localization_paused": localization_paused,
                "requested_target_dof_pos": np.asarray(
                    self._jsonable(requested_target_dof_pos)
                ).reshape(-1),
                "applied_target_dof_pos": np.asarray(
                    self._jsonable(applied_target_dof_pos)
                ).reshape(-1),
                "policy_action": np.asarray(
                    self._jsonable(policy_action)
                ).reshape(-1),
                "root_pos": self.mujoco_data.qpos[:3].copy(),
                "root_quat_wxyz": self.mujoco_data.qpos[3:7].copy(),
                "dof_pos": self.mujoco_data.qpos[self.body_qpos_adr].copy(),
                "dof_vel": self.mujoco_data.qvel[self.body_dof_adr].copy(),
                "applied_body_torque": self.last_body_torque.copy(),
                "reference_sample": reference_sample,
                "policy_observation": policy_observation,
            },
        )

    def get_hand_motor_state(self):
        """Return the commanded MuJoCo hand pose mapped to six physical motors."""
        import sys
        mapping_path = str(self.cfg.get("hand_mapping_path", "/home/nerv/qingyaoxu/TWIST2/deploy_real"))
        if mapping_path not in sys.path:
            sys.path.insert(0, mapping_path)
        from dofpos2cmd import dofpos12_to_q6
        # IK order differs from the TWIST2 lookup order; keep the same mapping
        # already audited for the real DDS worker.
        order = np.array([8, 9, 10, 11, 0, 1, 2, 3, 6, 7, 4, 5])
        return (
            np.asarray(dofpos12_to_q6(self.left_hand_target[order]), dtype=np.float32),
            np.asarray(dofpos12_to_q6(self.right_hand_target[order]), dtype=np.float32),
        )

    def refresh_sim(self):
        return {
            "root_pos": self._tensor(self.mujoco_data.qpos[:3]),
            "root_quat_wxyz": self._tensor(self.mujoco_data.qpos[3:7]),
            "base_ang_vel": self._tensor(self.mujoco_data.qvel[3:6]),
            "dof_pos": self._tensor(self.mujoco_data.qpos[self.body_qpos_adr]),
            "dof_vel": self._tensor(self.mujoco_data.qvel[self.body_dof_adr]),
        }

    @staticmethod
    def _tensor(value):
        import torch
        return torch.from_numpy(np.asarray(value)).float()

    def calibrate(self, init_state_dict={}):
        self.mujoco_data.qpos[:] = self.default_qpos
        self.mujoco_data.qvel[:] = self.default_qvel
        if "root_pos" in init_state_dict:
            self.mujoco_data.qpos[:3] = init_state_dict["root_pos"]
        if "root_quat" in init_state_dict:
            self.mujoco_data.qpos[3:7] = init_state_dict["root_quat"]
        if "dof_pos" in init_state_dict:
            self.mujoco_data.qpos[self.body_qpos_adr] = init_state_dict["dof_pos"]
        if "root_lin_vel" in init_state_dict:
            self.mujoco_data.qvel[:3] = init_state_dict["root_lin_vel"]
        if "root_ang_vel" in init_state_dict:
            self.mujoco_data.qvel[3:6] = init_state_dict["root_ang_vel"]
        if "dof_vel" in init_state_dict:
            self.mujoco_data.qvel[self.body_dof_adr] = init_state_dict["dof_vel"]
        self.mujoco_data.ctrl[:] = 0.0
        self.hand_elapsed = 0.0
        self._set_hand_pose(0.0)
        mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        return self.mujoco_data.qpos[:3].copy(), self.mujoco_data.qpos[3:7].copy()

    def apply_action(self, tgt_dof_pos):
        target = np.asarray(tgt_dof_pos).squeeze()
        for _ in range(self.decimation):
            body_torque = (
                (target - self.mujoco_data.qpos[self.body_qpos_adr]) * self.stiffness
                - self.mujoco_data.qvel[self.body_dof_adr] * self.damping
            )
            self.last_body_torque = body_torque.copy()
            self.mujoco_data.ctrl[:] = 0.0
            self.mujoco_data.ctrl[self.body_actuator_ids] = body_torque
            mujoco.mj_step(self.mujoco_model, self.mujoco_data)
            self.hand_elapsed += self.low_dt
            self._set_hand_pose(self.hand_elapsed)
            mujoco.mj_forward(self.mujoco_model, self.mujoco_data)
        self._render()
