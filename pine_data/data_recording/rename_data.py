import os

data_path = '/media/mainuser/a6300fe1-151f-4e9e-8790-c4826f4ee765/data_recording/data_weighing_scale_YuanYao/remove_vegetable_after_off'  # path of collected demonstrations


all_folders = sorted([
    item for item in os.listdir(data_path)
    if os.path.isdir(os.path.join(data_path, item))
])


temp_prefix = '__temp__'
for i, old_name in enumerate(all_folders):
    old_path = os.path.join(data_path, old_name)
    temp_path = os.path.join(data_path, f"{temp_prefix}{i}")
    os.rename(old_path, temp_path)


for i in range(len(all_folders)):
    temp_path = os.path.join(data_path, f"{temp_prefix}{i}")
    new_path = os.path.join(data_path, f"episode_{i}")
    os.rename(temp_path, new_path)
    print(f"[RENAME] {temp_path} → {new_path}")

print(f"\n✅ Renamed {len(all_folders)} folders to episode_0 ~ episode_{len(all_folders)-1}")
