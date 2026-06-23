To activate/deactivate venv
```
source ~/UR5_Policy/data_record_env/bin/activate
deactivate
```

To use spacemouse
```
cd spacemouse_teleoperation/
python 3DConnexion_UR5_Teleop_Gripper.py
```
## Data Collection
To collect data, run command in another terminal
```
cd data_recording/
python record_data.py
```
Give it some time to load, and you will be prompted to name the data. Afterwhich use the following commands.      
`c` to start recording    
`s` to stop & save recording    
`d` to delete most recent recording   
`q` to quit   


## Others
<!-- train stage -->
cd /home/mainuser/UR5_Policy/Schwarz_DP3

bash /home/mainuser/UR5_Policy/Schwarz_DP3/scripts/train_policy.sh dp3 <task_name> 0888 0 0



