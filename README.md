# Hand2Rob: Contact-Driven Learning from Humans

## Setup

### 1. NUC (Franka Controller)

SSH into the NUC and start the arm and gripper drivers:
```bash
# Terminal 1
tmux new -s franka_arm
cd ~/work/deoxys_control/deoxys
./auto_scripts/auto_arm.sh config/franka_left.yml

# Terminal 2
tmux new -s franka_gripper
cd ~/work/deoxys_control/deoxys
./auto_scripts/auto_gripper.sh config/franka_left.yml
```

### 2. Robot Workstation (robotoss)

```bash
cd ~/Point-Policy/Franka-Teach

# Terminal 1: Franka server
python3 franka_server.py

# Terminal 2: Camera streams
python3 camera_server.py

# Terminal 3: ResKin sensor (if collecting tactile data)
python3 reskin_server.py
```

### 3. Lambda Server

Clone and install:
```bash
git clone git@github.com:feel-the-force-ftf/feel-the-force.git --recurse-submodules
conda env create -f conda_env.yaml
conda activate force
bash setup.sh
```

Set up SSH tunnels from robotoss:
```bash
ssh -fN -o ExitOnForwardFailure=yes wsi3567@lamb.mech.northwestern.edu \
  -R 10006:127.0.0.1:10006 \
  -R 11006:127.0.0.1:11006 \
  -R 10007:127.0.0.1:10007 \
  -R 11007:127.0.0.1:11007 \
  -R 18901:127.0.0.1:8901 \
  -R 12005:127.0.0.1:12005
```

## Data Collection

```bash
cd ~/Point-Policy/Franka-Teach


# Terminal 1: Collect a demo (increment demo_num each time)
python3 collect_data.py storage_path=<storage_path> demo_num=0 collect_reskin=True
```

## Data Preprocessing

```bash
cd point-policy/robot_utils/franka

# 1. Process raw data
python process_data_human.py --data_dir <data_path> --task_names <task_name>

# 2. Generate calibration
cd calibration && python calibrate_apriltag_table.py

# 3. Convert to pkl (without points)
python convert_to_pkl_human.py --data_dir <data_path> --calib_path <calib_path> --task_names <task_name>

# 4. Label points in label_points.ipynb
ssh -L 8888:127.0.0.1:8888 wsi3567@lamb.mech.northwestern.edu

jupyter notebook --no-browser --port=8888

# 5. Convert to pkl (with points)
python convert_to_pkl_human.py --data_dir <data_path> --calib_path <calib_path>  --task_names <task_name> --process_points --smooth_tracks

# 6. Convert to robot actions
python convert_pkl_human_to_robot.py --data_dir <data_path> --calib_path <calib_path> --task_name <task_name> --smooth_robot_tracks
```

Set `data_dir` in `cfg/config.yaml` and `cfg/config_eval.yaml` to `path/to/data/expert_demos`.

## Training

```bash
cd point-policy

python train.py agent=point_policy suite=point_policy dataloader=point_policy \
    eval=false suite.use_robot_points=true suite.use_object_points=true \
    suite/task/franka_env=<task_name> experiment=point_policy \
    suite.predict_force=true
```

## Evaluation

```bash
# Model inference
python eval_point_track.py agent=point_policy suite=point_policy dataloader=point_policy \
    eval=true suite.use_robot_points=true suite.use_object_points=true \
    suite/task/franka_env=<task_name> experiment=eval_point_policy \
    bc_weight=/path/to/snapshot.pt suite.predict_force=true

# Replay demo
python eval_point_track.py agent=point_policy suite=point_policy dataloader=point_policy \
    eval=true suite.use_robot_points=true suite.use_object_points=true \
    suite/task/franka_env=<task_name> experiment=eval_point_policy \
    replay_demo=true suite.predict_force=true
```