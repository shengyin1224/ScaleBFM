# ScaleBridge

ScaleBridge provides an integrated Sim2Sim and Sim2Real deployment pipeline for Behavior Foundation Models on humanoid robots.

This guide covers workstation and Jetson setup, policy evaluation, policy deployment, and migration instructions. Run all commands from the ScaleBridge root unless stated otherwise.

> [!IMPORTANT]
> Real-robot deployment can cause sudden or unexpected motion. Keep an emergency stop within reach before launching the policy.

## 新 HAT checkpoint 如何接入 ScaleBFM

前提：HAT 和 ScaleBFM 各自已经可以独立运行。接入新 HAT checkpoint 时，
**ScaleBFM 模型和代码不需要重新编译**；只需要用新 checkpoint 启动 HAT
在线服务，再启动现有的 `hat4_online` ScaleBFM 配置。两个 Python 进程必须
运行在同一台工作站，因为当前默认通过本机端口 `5561/5562` 通信。

数据流只有一条：

```text
G1/Vive 当前状态 -> ScaleBFM -> HAT 在线推理 -> action chunk（默认 64 帧）-> ScaleBFM 跟踪 -> G1
```

### 1. 准备新 HAT checkpoint 的三个配套文件

新 checkpoint 必须配套使用它训练时的：

1. 模型 YAML，例如 `hdt/configs/models/hat_linear4.yaml`；
2. `dataset_stats.pkl`；
3. 已有相机回调 `module:function`。

YAML 必须与 checkpoint 的模型结构一致；stats 中的 `qpos_mean`、
`qpos_std`、`action_mean`、`action_std` 必须都是 128 维。HAT 输出宽度必须是
128，chunk 长度必须与 YAML 的 `action_chunk_size` 一致（未配置时默认 64），
并按训练时相同的 128 维语义包含 global root/head/wrist 和 wrist-local finger
XYZ。ACT TorchScript、ACT checkpoint 和 RDT state dict 均可由在线服务加载。

### 2. 启动顺序

Vive sender 和 `g1_29dof_controller` 仍按原来能够独立部署 ScaleBFM 的方式
启动，不需要改。然后依次打开下面两个终端。

终端 A：使用**新 HAT checkpoint**启动 HAT。把四个占位符替换为新模型的
实际路径和已有相机回调：

```bash
cd /home/nerv/qingyaoxu/human_policy
PYTHONPATH=/home/nerv/qingyaoxu/human_policy:/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge \
/home/nerv/miniconda3/envs/twist_qyx/bin/python -m hdt.online_hat_service \
  --config <新HAT模型.yaml> \
  --checkpoint <新HAT_checkpoint.pt或ckpt> \
  --stats <新HAT对应的dataset_stats.pkl> \
  --embodiment h1_inspire \
  --image-source <已有相机模块>:<get_images函数> \
  --fps 30 \
  --replan-frames 10
```

这个进程启动后等待 ScaleBFM 发来当前机器人状态；此时没有 chunk 日志是
正常的。`--replan-frames 10` 表示每约 0.33 秒基于最新状态重新生成一次 chunk。

终端 B：启动现有 ScaleBFM 真机部署，ScaleBFM checkpoint 不变：

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4_online \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  env.config.reference_forcing=false
```

### 3. 真机开始闭环

1. 按第一次 `R2`，完成 Vive root 标定；
2. 按第二次 `R2`，完成 ScaleBFM 与底层控制器通信；
3. ScaleBFM 开始把当前 global root、FK head/wrist、手臂关节和 Inspire
   实测手指状态发给 HAT；
4. HAT 用新 checkpoint 生成 chunk；ScaleBFM 终端应依次看到
   `Accepted chunk`、`Canonical FOCUS result attached` 和 `READY`；
5. 只有看到 `READY` 后才按 `R1`，开始 HAT -> ScaleBFM -> G1 实时闭环。

串联过程中，ScaleBFM 自动完成 chunk 的 global root 对齐、30 Hz 到 50 Hz
采样、FOCUS 两维 phase、finger XYZ 到 Inspire 的 IK，以及新旧 chunk 过渡；
不需要额外手工转换。超过 0.5 秒收不到 chunk 会冻结参考，超过 2 秒会锁住，
恢复新 chunk 后必须重新按一次 `R1`。

## 🧭 Table of contents

- [🛠️ 1. Prepare the environment](#️-1-prepare-the-environment)
- [📡 2. Configure optional tracking devices](#-2-configure-optional-tracking-devices)
- [🦾 3. Build the robot controller](#-3-build-the-robot-controller)
- [🎮 4. Launch the low-level controller](#-4-launch-the-low-level-controller)
- [🚀 5. Launch a policy](#-5-launch-a-policy)
- [💡 6. Examples](#-6-examples)
- [📦 7. Migrate the policy](#-7-migrate-the-policy)

## 🛠️ 1. Prepare the environment

Choose the setup instructions for the computer that will run the policy.

### Linux workstation

ScaleBridge require a **Python 3.11** environment on Linux workstation. (You may use the same environment with ScaleTrack.)

```bash
git clone https://github.com/zengweishuai/ScaleBFM.git
cd ScaleBFM/ScaleBridge
# Activate your environment with Python 3.11
pip install -e .
```

Install LCM development files for communication between the policy and the low-level robot controller:

```bash
sudo apt install liblcm-dev
```

Connect the workstation to the robot and assign its network interface an address in the `192.168.123.xxx` subnet, for example `192.168.123.222`. Verify connectivity with the G1 PC2 onboard computer:

```bash
ping 192.168.123.164
```

### NVIDIA Jetson (aarch64)

Clone ScaleBridge on the Jetson:

```bash
git clone https://github.com/zengweishuai/ScaleBFM.git
cd ScaleBFM/ScaleBridge
```

Update the device to JetPack 6.2 by following the [NVIDIA GR00T Whole-Body Control JetPack guide](https://nvlabs.github.io/GR00T-WholeBodyControl/references/jetpack6.html), then install JetPack and CUDA 12.6:

```bash
sudo apt-get update
sudo apt-get install nvidia-jetpack
sudo apt-get install cuda-toolkit-12-6
```

Install cuSPARSELt:

```bash
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-sbsa/libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
tar xf libcusparse_lt-linux-sbsa-0.5.2.1-archive.tar.xz
sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/include/. /usr/local/cuda/include/
sudo cp -a libcusparse_lt-linux-sbsa-0.5.2.1-archive/lib/. /usr/local/cuda/lib64/
```

Install LCM development files for communication between the policy and the low-level robot controller:

```bash
sudo apt install liblcm-dev
```

Choose one of the following methods to prepare the Python environment.

#### Option 1: Use the prebuilt Conda environment

Download the prebuilt aarch64 Conda environment, `scalebridge.tar.gz`, from [ScaleBFM on Hugging Face](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main/environment). Transfer the archive to the Jetson and extract it into the Conda environments directory:

```bash
scp scalebridge.tar.gz unitree@192.168.123.164:~/anaconda3/envs
ssh unitree@192.168.123.164
cd ~/anaconda3/envs
tar -xzf scalebridge.tar.gz
conda activate scalebridge
```

From the ScaleBridge root, run the editable installation to update the dependencies required by ScaleBridge and register the current checkout:

```bash
cd ~/ScaleBFM/ScaleBridge
pip install -e .
```

#### Option 2: Install from scratch

Activate a **Python 3.10** environment on the Jetson.

Install the JetPack-compatible PyTorch packages:

> [!WARNING]
> At the time this code was collected, the package source below appeared to be broken. This installation method may therefore not work as expected.

```bash
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

Install the bundled TensorRT and Torch-TensorRT wheels:

```bash
pip install dependencies/tensorrt-10.3.0-cp310-none-linux_aarch64.whl
pip install dependencies/torch_tensorrt-2.8.0+cu126-cp310-cp310-linux_aarch64.whl
```

Finally, install the ScaleBridge package:

```bash
pip install -e .
```

## 📡 2. Configure optional tracking devices

### Xsens for real-time teleoperation

1. Install and launch the Xsens software on a Windows computer.
2. Configure network streaming to the computer that runs the policy.
3. Select TCP and the default position-and-orientation data set.
4. Set the same port in Xsens and `env.config.xsens_port`.
5. If Xsens uses Manus gloves, enable them before streaming; hand data is included automatically.

### Vive Ultimate Tracker for root localization

1. Obtain [Vive Ultimate Trackers](https://www.vive.com/us/accessory/vive-ultimate-tracker/).
2. Print the pelvis connector from `accessories/vive_tracker_for_G1.stl`.
3. Launch Vive Hub and SteamVR, map the environment, and confirm that at least two trackers are available.
4. Place tracker 0 on the floor and attach tracker 1 to the G1 pelvis.
5. Check `localization_module.port` in the selected asset configuration (`scalebridge/config/asset/g1_29dof.yaml` or `scalebridge/config/asset/g1_29dof_dex3.yaml`). The default receiver port is `5000`.
6. Copy the `scalebridge/utils/vive_sender` directory to your Windows device.
7. Launch the tracker sender `tcp_send_raw_tracker_array.py` on the Windows computer using the policy computer's IP address and the same port as `localization_module.port`.

## 🦾 3. Build the robot controller

The robot controller serves as the bridge between the policy and the physical robot. During Sim2Real deployment, it reads the robot's sensor and joint state, forwards that state to the high-level policy, and applies the policy's joint commands to the robot. It must run alongside the high-level policy when using `simulator=real_world`; it is not needed for MuJoCo Sim2Sim evaluation.

### General command

Set `ROBOT_TYPE` to `g1_29dof` for the standard G1 or `g1_29dof_dex3` for the G1 with Dex3 hands, then build the matching controller.

```bash
cd third_party/unitree_sdk2
mkdir -p build
cd build
cmake .. -DROBOT_TYPE=ROBOT_TYPE
make
```

> [!IMPORTANT]
> Use `g1_29dof_dex3` only when you want to control the hands. Otherwise, use `g1_29dof` even if the robot is equipped with hands.

## 🎮 4. Launch the low-level controller

> [!WARNING]
> Suspend the robot securely before launching the controller. It automatically moves the robot to its default pose.

Use the same `ROBOT_TYPE` selected during compilation and pass the robot-facing network interface:

```bash
cd third_party/unitree_sdk2/build/bin
./"${ROBOT_TYPE}_controller" NETWORK_INTERFACE
```

On a Linux workstation, use the interface configured above. For onboard deployment, use the interface associated with the G1 PC2 address, `192.168.123.164`.

## 🚀 5. Launch a policy

Place each compiled checkpoint and its metadata file in the same model directory:

```text
scalebridge/data/model/robot_type
└── MODEL_NAME/
    ├── linux/
    │   ├── CHECKPOINT_tensorrt.pt
    │   └── CHECKPOINT_tensorrt_metadata.json
    └── aarch64/
        ├── CHECKPOINT_tensorrt.pt
        └── CHECKPOINT_tensorrt_metadata.json
```

The checkpoint and metadata filenames must share the same prefix. For example, `model_22200_tensorrt.pt` must be paired with `model_22200_tensorrt_metadata.json`.

> [!NOTE]
> Download the compiled models for Unitree G1 from [ScaleBFM on Hugging Face](https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main/compiled_checkpoint/), then place the checkpoint and metadata files in the structure shown above under `scalebridge/data/model/g1_29dof`. Each model includes a `linux` version compiled on an NVIDIA GeForce RTX 4090 Linux workstation and an `aarch64` version compiled onboard. Use the version that matches your deployment platform because TensorRT checkpoints are platform- and device-specific.

### General command

```bash
python scalebridge/run.py \
  agent.config.control_mode=MODE_INDEX \
  agent.config.checkpoint=/path/to/checkpoint.pt \
  asset=ROBOT_TYPE \
  env=TASK \
  env.config.reference_forcing=REFERENCE_FORCING \
  simulator=BACKEND
```

The command is composed from four Hydra configuration groups:

| Group | Purpose | Available values | Default |
| --- | --- | --- | --- |
| `agent` | Loads the BFM checkpoint and performs inference | `bfm_agent` | `bfm_agent` |
| `asset` | Selects the robot embodiment | `g1_29dof`, `g1_29dof_dex3` | `g1_29dof` |
| `env` | Selects the corresponding task | `motion_tracking`, `motion_tracking_xsens` | `motion_tracking` |
| `simulator` | Selects the execution backend | `mujoco_simulator`, `real_world` | `mujoco_simulator` |

> [!NOTE]
> Test every policy with `simulator=mujoco_simulator` before using `simulator=real_world`.
>
> Regardless of the selected control mode, the markers always show the whole-body target in global coordinates. When `reference_forcing=True`, they are not expected to align with the execution results.

### Configuration details

Here we provide a brief explanation of each configuration group. For more details, refer to the configuration files under `scalebridge/config`.

<details>
<summary><strong>🤖 Agent configuration</strong></summary>

Set `agent.config.checkpoint` to the compiled policy checkpoint. TensorRT artifacts are platform-specific: a checkpoint compiled on Linux x86_64 cannot be loaded directly on aarch64. Recompile it for the deployment platform by following the export instructions in the training codebase.

Set `agent.config.control_mode` to one of the following modes:

| Mode index | Control mode | Activated links |
| ---: | --- | --- |
| `0` | Root Mode | Pelvis |
| `1` | Bimanual Mode | Left Hand, Right Hand |
| `2` | Root-and-Hand Mode | Pelvis, Left Hand, Right Hand |
| `3` | End-Effector Mode | Left Hand, Right Hand, Left Foot, Right Foot |
| `4` | Root-and-End-Effector Mode | Pelvis, Left Hand, Right Hand, Left Foot, Right Foot |
| `5` | Upper-Body Mode | Left and Right Shoulders, Elbows, and Hands |
| `6` | Root-and-Upper-Body Mode | Pelvis, Left and Right Shoulders, Elbows, and Hands |
| `7` | Whole-Body Mode | Pelvis, Torso, Left and Right Shoulders, Elbows, Hands, Hips, Knees, and Feet |

ScaleBridge currently uses one fixed control mode for the full deployment session. Applications that require mode switching must implement that logic separately.

</details>

<details>
<summary><strong>🦾 Asset configuration</strong></summary>

The default asset is `g1_29dof`. Use the Dex3 configuration with:

```bash
asset=g1_29dof_dex3
```

When using `g1_29dof_dex3`, Manus gloves, and `env=motion_tracking_xsens`, also enable hand tracking:

```bash
env.config.has_hand=true
```

</details>

<details>
<summary><strong>🎯 Environment configuration</strong></summary>

The default option is `env=motion_tracking`, which replays an offline reference motion. Specify the motion file with `env.config.motion_path`

For real-time teleoperation, set `env=motion_tracking_xsens` and configure the Xsens receiver with `env.config.xsens_port`. ScaleBridge directly scales the human skeleton and does not include a separate retargeting pipeline. Adjust `env.config.xsens_scale_factor` to improve alignment.

Both tasks support local and global tracking through `env.config.reference_forcing`:

| Value | Tracking mode | Behavior |
| --- | --- | --- |
| `True` | Local | Uses the reference root position as the current root position |
| `False` | Global | Uses the root-localization module when it is available |

</details>

<details>
<summary><strong>🎮 Simulator configuration</strong></summary>

The simulator configuration controls execution timing and optional visualization features. Override an option from the command line with `simulator.config.OPTION=VALUE`.

| Option | Backend | Default | Purpose |
| --- | --- | --- | --- |
| `low_dt` | All | `0.005` | Low-level control timestep in seconds |
| `decimation` | All | `4` | Number of low-level steps per policy step |
| `record_video` | MuJoCo | `False` | Records `recording.mp4` in the Hydra output directory |
| `marker` | MuJoCo | `True` | Displays reference-body markers |
| `camera_follow` | MuJoCo | `True` | Keeps the camera centered on the robot |

For example, enable video recording during MuJoCo evaluation with:

```bash
simulator=mujoco_simulator simulator.config.record_video=True
```

</details>

## 💡 6. Examples

### Replay an offline motion in Whole-Body Mode with local tracking

1. Validate the compiled policy and reference motion with MuJoCo Sim2Sim on the Linux workstation:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=True \
     env.config.motion_path=/path/to/motion.npz \
     simulator=mujoco_simulator
   ```

2. Proceed only after confirming that the policy performs as expected in simulation. Suspend the robot securely, then launch the low-level controller:

   ```bash
   cd third_party/unitree_sdk2/build/bin
   ./g1_29dof_controller NETWORK_INTERFACE
   ```

3. In a separate terminal, launch the policy with the real-world backend:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=True \
     env.config.motion_path=/path/to/motion.npz \
     simulator=real_world
   ```

4. Carefully lower the robot until its feet contact the ground and it reaches a stable standing configuration. If the balance point is difficult to locate, gradually release tension from the suspension system while keeping the robot supported.

5. Follow the instructions in the terminal and press `R2` once on the remote controller to calibrate the robot state.

6. After confirming that the robot and surrounding area are ready, press `R2` again to establish communication between the policy and the controller and begin execution.

> [!IMPORTANT]
> In any emergency, press `L2 + B`. A **single press** immediately switches the robot to damping mode and stops policy-command execution. A **double press** terminates the controller after it enters damping mode, which may cause the robot to suddenly lose actuator support and go limp.


### Replay an offline motion in Whole-Body Mode with global tracking

1. Validate the compiled policy and reference motion with MuJoCo Sim2Sim on the Linux workstation:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=False \
     env.config.motion_path=/path/to/motion.npz \
     simulator=mujoco_simulator
   ```

2. Proceed only after confirming that the policy performs as expected in simulation. Suspend the robot securely, then launch the low-level controller:

   ```bash
   cd third_party/unitree_sdk2/build/bin
   ./g1_29dof_controller NETWORK_INTERFACE
   ```

3. Connect the Vive Tracker and put the one with index 0 on the ground, with its logo up. Mount another one with index 1 on the robot's hip **in the same orientation shown in the figure**.

   <p align="center">
     <img src="assets/vive_install.png" alt="Vive Tracker mounted on the robot's hip">
   </p>


4. Copy the `scalebridge/utils/vive_sender` directory to your Windows device. This is independent from `scalebridge`.:
    ```bash
    pip install argparse openvr
    cd vive_sender/
    python tcp_send_raw_tracker_array.py <deploy_device_ip> --port <deploy_device_port>
    ```

5. In a separate terminal, launch the policy with the real-world backend:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking \
     env.config.reference_forcing=False \
     env.config.motion_path=/path/to/motion.npz \
     simulator=real_world
   ```

6. Carefully lower the robot until its feet contact the ground and it reaches a stable standing configuration. If the balance point is difficult to locate, gradually release tension from the suspension system while keeping the robot supported.

7. Follow the instructions in the terminal and press `R2` once on the remote controller to calibrate the robot state. Ensure the robot stay still and the Vive Tracker faces directly to the bottom in a well-aligned pose.

8. After confirming that the robot and surrounding area are ready, press `R2` again to establish communication between the policy and the controller and begin execution.

### Teleoperate the robot with Xsens

1. Validate the compiled policy and reference motion with MuJoCo Sim2Sim on the Linux workstation:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking_xsens \
     env.config.reference_forcing=True \
     simulator=mujoco_simulator
   ```

2. Proceed only after confirming that the policy performs as expected in simulation. Suspend the robot securely, then launch the low-level controller:

   ```bash
   cd third_party/unitree_sdk2/build/bin
   ./g1_29dof_controller NETWORK_INTERFACE
   ```

3. In a separate terminal, launch the policy with the real-world backend:

   ```bash
   python scalebridge/run.py \
     agent.config.control_mode=7 \
     agent.config.checkpoint=/path/to/checkpoint.pt \
     asset=g1_29dof \
     env=motion_tracking_xsens \
     env.config.reference_forcing=True \
     simulator=real_world
   ```

4. Carefully lower the robot until its feet contact the ground and it reaches a stable standing configuration. If the balance point is difficult to locate, gradually release tension from the suspension system while keeping the robot supported.

5. Follow the terminal prompts and press `R2` once on the remote controller to calibrate the robot state. During calibration, the human operator should stand upright, preferably facing the same direction as the robot.

6. After confirming that the robot and surrounding area are ready, press `R2` again to establish communication between the policy and the controller and begin execution.


## 📦 7. Migrate the policy

The exported policy consumes raw sensor observations, so it can be integrated into another deployment codebase.

### Load the policy

Confirm that compatible versions of PyTorch and Torch-TensorRT are installed, then load the checkpoint:

```python
import torch
import torch_tensorrt

policy = torch.jit.load(ckpt_path).to("cuda")
policy.eval()
```

If the checkpoint and runtime versions are incompatible, recompile the checkpoint by following the training codebase instructions.

### Run inference

```python
tgt_dof_pos, action = policy(
    root_quat_buffer,
    base_ang_vel_buffer,
    dof_pos_buffer,
    dof_vel_buffer,
    actions_buffer,
    target_body_pos_future_to_robot_base,
    target_body_rot_future_to_robot_base,
    control_mode,
    future_time_offsets,
)
```

The policy inputs are:

| Input | Shape | Description |
| --- | --- | --- |
| `root_quat_buffer` | `(1, 3, 4)` | Pelvis IMU quaternion history in `wxyz` order |
| `base_ang_vel_buffer` | `(1, 3, 3)` | Pelvis IMU angular-velocity history in `xyz` order |
| `dof_pos_buffer` | `(1, 3, num_joints)` | Raw joint-position history; order follows metadata `joint_names` |
| `dof_vel_buffer` | `(1, 3, num_joints)` | Raw joint-velocity history; order follows metadata `joint_names` |
| `actions_buffer` | `(1, 3, num_actions)` | Raw action history; order follows metadata `action_names` |
| `target_body_pos_future_to_robot_base` | `(1, 6, 14, 3)` | Target body positions expressed in the root frame |
| `target_body_rot_future_to_robot_base` | `(1, 6, 14, 4)` | Target body rotations in `wxyz` order, expressed in the root frame |
| `control_mode` | `(1,)` | Control-mode index |
| `future_time_offsets` | `(6,)` | Time offsets corresponding to the target-body future frames |

For both target-body tensors, link order follows metadata `selected_body_names`. The six future frames are `[0, 1, 2, 3, 4, X]`, where `X` can be any step from 5 through 33 and each step represents `0.02 s`.

The policy returns:

| Output | Shape | Description |
| --- | --- | --- |
| `tgt_dof_pos` | `(1, num_actions)` | Target joint positions after action scale and offset are applied |
| `action` | `(1, num_actions)` | Raw actions before scale and offset are applied |

Both output orders follow metadata `action_names`. The checkpoint metadata also contains:

| Key | Description |
| --- | --- |
| `stiffness` | Execution stiffness values in `action_names` order |
| `damping` | Execution damping values in `action_names` order |
