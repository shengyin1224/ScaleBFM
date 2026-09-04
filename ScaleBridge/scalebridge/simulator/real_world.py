import sys
sys.path.append("./")

import lcm
import time
import torch
import select
import threading
import subprocess
import os
import json
import atexit
import copy
import queue
import numpy as np
from loguru import logger
from hydra.utils import instantiate
from scalebridge.simulator.base_simulator import BaseSimulator

class RealWorld(BaseSimulator):
    
    def __init__(self, config, metadata_dict):
        self.use_joystick = config.get('joystick', False)
        super().__init__(config, metadata_dict)
        self._init_communication()

    def _setup_backbone(self):
        super()._setup_backbone()
        self.lcm = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
    
    def _setup_asset(self):
        super()._setup_asset()
        
        # remote controller metadata
        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0
        self.commands_tmp = np.zeros((3,), dtype=np.float32)

        self.rc_decoder = instantiate(self.cfg.asset.rc_decoder)
        self.state_decoder = instantiate(self.cfg.asset.state_decoder)
        self.command_encoder = instantiate(self.cfg.asset.command_encoder)

        if "enable_root_localization" in self.metadata_dict and self.metadata_dict["enable_root_localization"]:
            robust = bool(self.cfg.get("tracker_robust_mode", False))
            self.localization_module = instantiate(
                self.cfg.asset.localization_module,
                robust_tracking=robust,
                stale_timeout=float(self.cfg.get("tracker_stale_timeout", 0.25)),
                recovery_valid_frames=int(self.cfg.get("tracker_recovery_valid_frames", 10)),
                calibration_frames=int(self.cfg.get("tracker_calibration_frames", 50)),
                max_position_step=float(self.cfg.get("tracker_max_position_step", 0.08)),
                max_position_speed=float(self.cfg.get("tracker_max_position_speed", 3.0)),
                position_smoothing_alpha=float(self.cfg.get("tracker_position_smoothing_alpha", 0.2)),
                calibration_max_position_std=float(self.cfg.get("tracker_calibration_max_position_std", 0.01)),
            )
        else:
            self.localization_module = None

        self._voice_process = None
        self._voice_lock = threading.Lock()
        self._voice_reader_thread = None
        self._voice_enabled = bool(self.cfg.get("tracker_robust_mode", False)) and bool(
            self.cfg.get("tracker_voice_alerts", True)
        )
        if self._voice_enabled:
            self._start_voice_worker()
        # Robot-state watchdog. On 2026-08-17 the robot NIC link dropped twice
        # mid-motion; the policy kept publishing targets computed against a
        # frozen state, and when the link came back the robot received targets
        # up to 2.6 rad away from its actual pose. Real sensor streams always
        # jitter, so a bit-identical state for longer than the hold timeout
        # means the link to the robot is dead, not that the robot is still.
        self._watchdog_enabled = bool(self.cfg.get("state_watchdog", True))
        self._watchdog_hold_timeout = float(self.cfg.get("state_watchdog_hold_timeout", 0.25))
        self._watchdog_damping_timeout = float(self.cfg.get("state_watchdog_damping_timeout", 1.0))
        self._last_state_change_mono = None
        self._watchdog_holding = False
        self._watchdog_latched = False
        self._watchdog_last_log_mono = 0.0
        self._deployment_record = None
        self._deployment_record_lock = threading.Lock()
        self._last_deployment_record_flush_mono = 0.0
        self._hat_chunk_record = None
        self._hat_chunk_record_lock = threading.Lock()
        self._hat_chunk_record_queue = queue.Queue(maxsize=64)
        self._hat_chunk_record_thread = None
        self._hat_chunk_record_closed = False
        self._hat_chunk_record_failed = False
        self._last_hat_chunk_record_flush_mono = 0.0
        atexit.register(self._close_deployment_record)
        atexit.register(self._close_hat_chunk_record)
        atexit.register(self._close_voice_worker)

    def _start_deployment_record(self):
        if not self.robust_tracking_enabled() or self._deployment_record is not None:
            return
        try:
            from hydra.core.hydra_config import HydraConfig
            output_dir = HydraConfig.get().runtime.output_dir
        except (ImportError, ValueError):
            output_dir = os.getcwd()
        path = os.path.abspath(os.path.join(output_dir, "real_world_control.jsonl"))
        self._deployment_record = open(path, "w", encoding="utf-8")
        self._last_deployment_record_flush_mono = time.monotonic()
        self._deployment_record.write(json.dumps({
            "record_type": "metadata", "started_at": time.time(),
            "fields": ["policy_action", "target_dof_pos", "dof_pos", "dof_vel",
                       "root_quat_wxyz", "base_ang_vel", "global_root_pos",
                       "localization_healthy", "localization_paused"],
        }, separators=(",", ":")) + "\n")
        self._deployment_record.flush()
        logger.info(f"[Simulator] Recording real-world controls to {path}")

    def _close_deployment_record(self):
        with self._deployment_record_lock:
            stream, self._deployment_record = self._deployment_record, None
            if stream is not None:
                try:
                    stream.flush()
                    stream.close()
                except OSError:
                    pass

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
            return {key: RealWorld._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [RealWorld._jsonable(item) for item in value]
        raise TypeError(
            f"Cannot serialize HAT chunk value of type {type(value).__name__}"
        )

    def _hat_chunk_record_loop(self):
        try:
            try:
                from hydra.core.hydra_config import HydraConfig
                output_dir = HydraConfig.get().runtime.output_dir
            except (ImportError, ValueError):
                output_dir = os.getcwd()
            path = os.path.abspath(
                os.path.join(output_dir, "online_hat_chunks.jsonl")
            )
            self._hat_chunk_record = open(path, "w", encoding="utf-8")
            self._last_hat_chunk_record_flush_mono = time.monotonic()
            logger.info(
                f"[Simulator] Recording complete online HAT chunks to {path}"
            )
            while True:
                record = self._hat_chunk_record_queue.get()
                try:
                    if record is None:
                        return
                    self._hat_chunk_record.write(
                        json.dumps(
                            self._jsonable(record), separators=(",", ":")
                        ) + "\n"
                    )
                    now_mono = time.monotonic()
                    if now_mono - self._last_hat_chunk_record_flush_mono >= 1.0:
                        self._hat_chunk_record.flush()
                        self._last_hat_chunk_record_flush_mono = now_mono
                finally:
                    self._hat_chunk_record_queue.task_done()
        except Exception as exc:
            self._hat_chunk_record_failed = True
            logger.error(f"[Simulator] Online HAT chunk recording failed: {exc}")
        finally:
            stream, self._hat_chunk_record = self._hat_chunk_record, None
            if stream is not None:
                try:
                    stream.flush()
                    stream.close()
                except OSError:
                    pass

    def _close_hat_chunk_record(self):
        with self._hat_chunk_record_lock:
            if self._hat_chunk_record_closed:
                return
            self._hat_chunk_record_closed = True
            thread = self._hat_chunk_record_thread
            if thread is None:
                return
            self._hat_chunk_record_queue.put(None)
        thread.join(timeout=5.0)
        if thread.is_alive():
            logger.warning(
                "[Simulator] Timed out flushing the online HAT chunk record."
            )

    def record_hat_chunk(self, raw_chunk, aligned_chunk):
        """Record every complete accepted HAT chunk before and after alignment."""
        with self._hat_chunk_record_lock:
            if self._hat_chunk_record_closed or self._hat_chunk_record_failed:
                return
            if self._hat_chunk_record_thread is None:
                self._hat_chunk_record_thread = threading.Thread(
                    target=self._hat_chunk_record_loop,
                    name="hat-chunk-recorder",
                    daemon=True,
                )
                self._hat_chunk_record_thread.start()

            record = {
                "record_type": "hat_chunk",
                "time": time.time(),
                "monotonic": time.monotonic(),
                # The aligned chunk later receives FOCUS in another worker.
                # Snapshot both dictionaries before handing them to the writer.
                "raw": copy.deepcopy(raw_chunk),
                "aligned": copy.deepcopy(aligned_chunk),
            }
            try:
                self._hat_chunk_record_queue.put_nowait(record)
            except queue.Full:
                # Never block the real-time policy loop on diagnostic I/O.
                logger.error(
                    "[Simulator] Online HAT chunk record queue is full; "
                    "dropping this diagnostic chunk."
                )

    def _close_voice_worker(self):
        process, self._voice_process = self._voice_process, None
        if process is None:
            return
        try:
            if process.stdin is not None:
                process.stdin.close()
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()
        except OSError:
            pass

    def _record_control_frame(self, target_dof_pos, policy_action=None, localization_paused=None):
        with self._deployment_record_lock:
            stream = self._deployment_record
            if stream is None:
                return
            root_pos = None
            if self.localization_module is not None:
                root_pos = self.localization_module.get_root_pos().tolist()
            record = {
                "record_type": "control_frame",
                "time": time.time(),
                "monotonic": time.monotonic(),
                "target_dof_pos": np.asarray(target_dof_pos).reshape(-1).tolist(),
                "policy_action": (
                    None if policy_action is None
                    else np.asarray(policy_action).reshape(-1).tolist()
                ),
                "dof_pos": self.dof_pos_tmp.copy()[self.sim_to_env_joint_idx].tolist(),
                "dof_vel": self.dof_vel_tmp.copy()[self.sim_to_env_joint_idx].tolist(),
                "root_quat_wxyz": self.root_quat_tmp.tolist(),
                "base_ang_vel": self.base_ang_vel_tmp.tolist(),
                "global_root_pos": root_pos,
                "localization_healthy": self.is_localization_healthy(),
                "localization_paused": localization_paused,
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            now_mono = record["monotonic"]
            if now_mono - self._last_deployment_record_flush_mono >= 1.0:
                stream.flush()
                self._last_deployment_record_flush_mono = now_mono

    def _start_voice_worker(self):
        worker = os.path.join(os.path.dirname(__file__), "g1_voice_worker.py")
        python = str(self.cfg.get("tracker_voice_python", "/home/nerv/miniconda3/envs/twist2/bin/python"))
        sdk_path = str(self.cfg.get("tracker_voice_sdk_path", "/home/nerv/qingyaoxu/TWIST2/unitree_sdk2_python"))
        net = str(self.cfg.get("tracker_voice_network", "enp4s0"))
        volume = int(self.cfg.get("tracker_voice_volume", 25))
        # Speaker 0 is the voice the on-robot Inspire service uses for Chinese;
        # speaker 1 reads Chinese text as gibberish.
        speaker_id = int(self.cfg.get("tracker_voice_speaker_id", 0))
        try:
            self._voice_process = subprocess.Popen(
                [python, "-u", worker, "--net", net, "--sdk-path", sdk_path,
                 "--volume", str(volume), "--speaker-id", str(speaker_id)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            self._voice_reader_thread = threading.Thread(
                target=self._read_voice_worker_output,
                args=(self._voice_process,),
                daemon=True,
                name="TrackerVoiceOutput",
            )
            self._voice_reader_thread.start()
            logger.info(
                f"[Simulator] Tracker voice alert worker started "
                f"(volume={max(0, min(100, volume))}, speaker_id={speaker_id})."
            )
        except Exception as exc:
            self._voice_process = None
            logger.warning(f"[Simulator] Tracker voice alert unavailable: {exc}")

    @staticmethod
    def _read_voice_worker_output(process):
        try:
            for line in process.stdout:
                message = line.strip()
                if message:
                    level = logger.info if message.startswith(("READY", "TTS_OK")) else logger.warning
                    level(f"[Tracker voice worker] {message}")
        except (OSError, ValueError):
            pass

    def _speak_tracker_event(self, event_name):
        text = {
            "tracker_lost": "腰部 Tracker 断联",
            "tracker_reconnected": "腰部 Tracker 已连接",
            "watchdog_damping_latched": "机器人通信超时,已进入阻尼保护",
            "motion_playback_complete": "动作完成,进入原地站立保持",
        }.get(event_name)
        if not text or self._voice_process is None or self._voice_process.poll() is not None:
            return
        with self._voice_lock:
            try:
                self._voice_process.stdin.write(text + "\n")
                self._voice_process.stdin.flush()
                logger.info(f"[Simulator] Tracker voice alert queued: {text}")
            except (BrokenPipeError, OSError):
                pass

    def _poll_localization_events(self):
        module = self.localization_module
        if module is None or not hasattr(module, "consume_tracking_events"):
            return
        for event_name, _ in module.consume_tracking_events():
            logger.warning(f"[Localization] {event_name}")
            self._speak_tracker_event(event_name)

    def _get_joint_names(self):
        return self.cfg.asset.real_joint_names # weishuai: FIXME maybe it should read from SDK or something else instead of manual assignment

    def _init_communication(self):
        self.firstReceiveRobotState = False
        self.firstReceiveRemoteController = False

        self._init_time = time.time()
        
        self.robot_state_subscriber = self.lcm.subscribe('robot_state_data', self._robot_state_handler)
        self.remote_controller_subscriber = self.lcm.subscribe('rc_command_data', self._remote_controller_handler)
        
        # Do not let the perpetual LCM polling loop keep the process alive
        # after the main policy loop receives Ctrl+C.
        self.run_thread = threading.Thread(target=self._poll, daemon=True)
        self.run_thread.start()

        logger.info(f"[Simulator] Waiting for the first robot state signal to arrive ...")
        while not self.firstReceiveRobotState:
            time.sleep(1)          

        if self.localization_module:
            self.localization_module.listen()
            logger.info(f"[Simulator] Waiting for the localization module client ...")  
            addr = self.localization_module.accept_blocking()
            logger.info(f"[Simulator] Localization module client connected: {addr}")
            self.localization_module.start()
            while not len(self.localization_module.buffer) > 0:
                time.sleep(0.02)
            logger.info(f"[Simulator] First signal from localization module arrives.")
    
    def _robot_state_handler(self, channel, data):
        robot_state = self.state_decoder.decode(data)

        dof_pos = np.array(robot_state.q)
        dof_vel = np.array(robot_state.qd)
        base_ang_vel = np.array(robot_state.omegaBody)
        root_quat = np.array(robot_state.quat)
        # Freshness for the watchdog counts only frames whose content changed:
        # the transition layer may keep republishing the last DDS sample after
        # the robot link dies, and re-arrivals of frozen data are not news.
        if (
            self._last_state_change_mono is None
            or not np.array_equal(dof_pos, self.dof_pos_tmp)
            or not np.array_equal(dof_vel, self.dof_vel_tmp)
            or not np.array_equal(base_ang_vel, self.base_ang_vel_tmp)
            or not np.array_equal(root_quat, self.root_quat_tmp)
        ):
            self._last_state_change_mono = time.monotonic()

        self.dof_pos_tmp = dof_pos
        self.dof_vel_tmp = dof_vel
        self.base_ang_vel_tmp = base_ang_vel
        self.root_quat_tmp = root_quat

        if not self.firstReceiveRobotState:
            self.time_delay = time.time() - self._init_time
            self.firstReceiveRobotState = True
            logger.info('[Simulator] Communication build successfully between the policy and the transition layer!')
            logger.info(f'[Simulator] First signal arrives after {self.time_delay}s!')
    
    def _remote_controller_handler(self, channel, data):
        msg = self.rc_decoder.decode(data)
        if not self.firstReceiveRemoteController:
            self.firstReceiveRemoteController = True
            logger.info('[Simulator] Communication build successfully between the policy and the remote controller!')
        
        self.left_upper_switch_pressed = ((msg.left_upper_switch and not self.left_upper_switch) or self.left_upper_switch_pressed)
        self.left_lower_left_switch_pressed = ((msg.left_lower_left_switch and not self.left_lower_left_switch) or self.left_lower_left_switch_pressed)
        self.left_lower_right_switch_pressed = ((msg.left_lower_right_switch and not self.left_lower_right_switch) or self.left_lower_right_switch_pressed)
        self.right_upper_switch_pressed = ((msg.right_upper_switch and not self.right_upper_switch) or self.right_upper_switch_pressed)
        self.right_lower_left_switch_pressed = ((msg.right_lower_left_switch and not self.right_lower_left_switch) or self.right_lower_left_switch_pressed)
        self.right_lower_right_switch_pressed = ((msg.right_lower_right_switch) and not self.right_lower_right_switch) or self.right_lower_right_switch_pressed

        self.mode = msg.mode
        self.right_stick = msg.right_stick
        self.left_stick = msg.left_stick
        self.left_upper_switch = msg.left_upper_switch
        self.left_lower_left_switch = msg.left_lower_left_switch
        self.left_lower_right_switch = msg.left_lower_right_switch
        self.right_upper_switch = msg.right_upper_switch
        self.right_lower_left_switch = msg.right_lower_left_switch
        self.right_lower_right_switch = msg.right_lower_right_switch

        self.commands_tmp = np.array([
            self.left_stick[1],
            self.left_stick[0] * -1,
            self.right_stick[0] * -1
        ], dtype=np.float32)
        self.commands_tmp = np.where(np.abs(self.commands_tmp) < 0.05, 0, self.commands_tmp)

    def _poll(self, cb=None):
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lcm.fileno()], [], [], timeout)
                if rfds:
                    self.lcm.handle()
                else:
                    continue
        except KeyboardInterrupt:
            pass

    def is_localization_healthy(self):
        # A stale robot state means the link to the robot is down: freeze the
        # reference exactly like a tracker outage, so the published targets do
        # not run ahead of a robot that cannot hear them.
        if self._watchdog_enabled and (
            self._watchdog_latched
            or self._robot_state_age() > self._watchdog_hold_timeout
        ):
            return False
        module = self.localization_module
        if module is None or not hasattr(module, "is_tracking_healthy"):
            return True
        return bool(module.is_tracking_healthy())

    def _robot_state_age(self):
        if self._last_state_change_mono is None:
            return 0.0
        return time.monotonic() - self._last_state_change_mono

    def robust_tracking_enabled(self):
        return bool(self.cfg.get("tracker_robust_mode", False)) and self.localization_module is not None

    def refresh_sim(self):

        self._poll_localization_events()

        state_dict = {
            "root_quat_wxyz": self.root_quat_tmp.copy(),
            "base_ang_vel": self.base_ang_vel_tmp.copy(),
            "dof_pos": self.dof_pos_tmp.copy()[self.sim_to_env_joint_idx],
            "dof_vel": self.dof_vel_tmp.copy()[self.sim_to_env_joint_idx],
        }

        if self.localization_module:
            state_dict.update({"root_pos": self.localization_module.get_root_pos()})

        if self.use_joystick:
            state_dict.update({"commands": self.commands_tmp.copy()})

        return {k:torch.from_numpy(v).float() for k,v in state_dict.items()}
    
    def calibrate(self, init_state_dict={}):
        assert not init_state_dict, f"Current code does not support reference state initialization for real-world deployment."

        logger.info('[Simulator] Calibraiting..., Press R2 to continue')
        while True:
            if self.right_lower_right_switch_pressed:
                logger.info('[Simulator] R2 button pressed, Start Calibrating...')
                self.right_lower_right_switch_pressed = False
                if self.localization_module and hasattr(self.localization_module, 'start_recording'):
                    self.localization_module.start_recording()
                    self.localization_module.record_event('first_r2_calibration')
                self._start_deployment_record()
                break
        
        root_quat = self.root_quat_tmp.copy()
        if self.localization_module:
            uncalibrated_root_pos = self.localization_module.get_root_pos()
            root_height = float(uncalibrated_root_pos[2])
            expected_root_height = float(self.cfg.asset.root_height)
            height_tolerance = float(self.cfg.get('localization_root_height_tolerance', 0.30))
            logger.info(
                f'[Simulator] Localization root height before calibration: {root_height:.3f}m '
                f'(expected {expected_root_height:.3f}m ± {height_tolerance:.3f}m)'
            )
            if not np.isfinite(root_height) or abs(root_height - expected_root_height) > height_tolerance:
                raise RuntimeError(
                    f'Unsafe localization root height {root_height:.3f}m; expected '
                    f'{expected_root_height:.3f}m ± {height_tolerance:.3f}m. '
                    'Check tracker order, visibility, and coordinate conversion before deployment.'
                )
            self.localization_module.calibrate(root_quat)
            root_pos = self.localization_module.get_root_pos()
        else:
            root_pos = np.zeros((3,), dtype=np.float32)
        
        logger.info('[Simulator] Calibration Done. Press R2 to continue')
        while True:
            if self.right_lower_right_switch_pressed:
                if self.cfg.get('reference_play_gate', False):
                    logger.info(
                        '[Simulator] R2 pressed again. The policy will hold the initial '
                        'reference pose; press R1 to start motion tracking.'
                    )
                else:
                    logger.info('[Simulator] R2 pressed again, Communication built between policy layer and transition layer!')
                self.right_lower_right_switch_pressed =  False
                self.record_localization_event('second_r2_policy_ready')
                break

        # Discard an R1 edge that may have been generated while calibration was
        # still in progress. Motion playback must require a fresh R1 press after
        # the policy has entered its ready/standing state.
        self.right_upper_switch_pressed = False
        return root_pos, root_quat

    def consume_reference_play_request(self):
        """Consume one rising-edge R1 request used to start reference playback."""
        if self.right_upper_switch_pressed:
            self.right_upper_switch_pressed = False
            return True
        return False

    def record_localization_event(self, name):
        if self.localization_module and hasattr(self.localization_module, 'record_event'):
            self.localization_module.record_event(name)

    def _watchdog_allows_publish(self):
        """Gate policy targets on robot-state freshness.

        Stage 1 (hold): past ``state_watchdog_hold_timeout`` stop publishing;
        the transition layer keeps forwarding the last pre-outage target, whose
        error is bounded by the outage detection time. Auto-releases when data
        resumes, because the reference was frozen meanwhile.

        Stage 2 (damping latch): past ``state_watchdog_damping_timeout`` the
        outage is real. Publish a pure-damping command every cycle so the
        FIRST thing the robot executes when the link recovers is damping, not
        a target computed against a frozen state. The latch is permanent:
        restart the program to run again.
        """
        age = self._robot_state_age()
        now = time.monotonic()
        if self._watchdog_latched or age > self._watchdog_damping_timeout:
            if not self._watchdog_latched:
                self._watchdog_latched = True
                self.record_localization_event("watchdog_damping_latched")
                logger.error(
                    f"[Watchdog] Robot state frozen for {age:.2f}s: link to the "
                    "robot is considered dead. Publishing damping and latching; "
                    "restart the program after fixing the connection."
                )
                self._speak_tracker_event("watchdog_damping_latched")
            self._publish_damping_command()
            return False
        if age > self._watchdog_hold_timeout:
            if not self._watchdog_holding:
                self._watchdog_holding = True
                self.record_localization_event("watchdog_hold")
            if now - self._watchdog_last_log_mono >= 1.0:
                logger.warning(
                    f"[Watchdog] Robot state stale for {age:.2f}s: holding the "
                    "last published target."
                )
                self._watchdog_last_log_mono = now
            return False
        if self._watchdog_holding:
            self._watchdog_holding = False
            self.record_localization_event("watchdog_hold_released")
            logger.warning(
                "[Watchdog] Robot state fresh again: resuming policy targets."
            )
        return True

    def _publish_damping_command(self):
        zeros = np.zeros((self.num_joints,), dtype=np.float32)
        cmd = self.command_encoder
        cmd.q_des = zeros.copy()
        cmd.qd_des = zeros.copy()
        cmd.kp = zeros.copy()
        cmd.kd = self.damping.copy()
        cmd.tau_ff = zeros.copy()
        cmd.se_contactState = np.zeros(2)
        cmd.timestamp_us = int(time.time()*10**6)
        self.lcm.publish(f"pd_plustau_targets", cmd.encode())

    def apply_action(self, tgt_dof_pos):
        if self._watchdog_enabled and not self._watchdog_allows_publish():
            return
        tgt_dof_pos = tgt_dof_pos.squeeze()

        target_dof_pos_in_sim = np.zeros((self.num_joints,), dtype=np.float32)
        target_dof_pos_in_sim[self.env_action_to_sim_idx] = tgt_dof_pos

        cmd = self.command_encoder
        cmd.q_des = target_dof_pos_in_sim.copy()
        cmd.qd_des = np.zeros_like(target_dof_pos_in_sim)
        cmd.kp = self.stiffness.copy()
        cmd.kd = self.damping.copy()
        cmd.tau_ff = np.zeros_like(target_dof_pos_in_sim)
        cmd.se_contactState = np.zeros(2)
        cmd.timestamp_us = int(time.time()*10**6)

        self.lcm.publish(f"pd_plustau_targets", cmd.encode())

    def record_policy_step(self, target_dof_pos, policy_action, localization_paused):
        self._record_control_frame(target_dof_pos, policy_action, localization_paused)
