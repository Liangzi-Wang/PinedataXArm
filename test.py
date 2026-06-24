from xarm.wrapper import XArmAPI
import time


class XArmController:
    """
    xArm 机械臂与 G2 夹爪控制器封装
    """

    def __init__(self, ip="192.168.1.206"):
        """
        初始化控制器并连接机械臂
        :param ip: 机械臂 IP 地址
        """
        self.arm = XArmAPI(ip)
        self._connect()

    def _connect(self):
        """内部方法：执行连接和基础状态检查"""
        if not self.arm.connected:
            print(f"无法连接到 {self.arm.client_ip}，请检查网络或IP。")
            return False

        # 清除报错
        self.arm.clean_error()
        self.arm.clean_warn()

        # 使能机械臂
        self.arm.motion_enable(enable=True)

        # 设置模式为位置控制 (0)
        self.arm.set_mode(0)

        # 设置状态为就绪 (0)
        self.arm.set_state(0)

        # 等待一小段时间确保指令生效
        time.sleep(1)

        # 检查是否处于无报错的就绪状态
        code, state = self.arm.get_state()
        if self.arm.warn_code == 0 and self.arm.error_code == 0:
            print("✅ xArm 连接成功，系统就绪。")
            return True
        else:
            print(f"❌ xArm 连接失败或存在报错 (warn_code: {self.arm.warn_code }), error_code: {self.arm.error_code }) 请检查硬件。")
            return False

    def move_relative(self, dx=0, dy=0, dz=0, dr=0, dp =0, dyaw=0, speed=200):
        """
        相对当前位置进行直线运动（上下左右前后）
        :param dx: X轴偏移量 (mm), 正数为右/前
        :param dy: Y轴偏移量 (mm), 正数为左/后
        :param dz: Z轴偏移量 (mm), 正数为上
        :param speed: 运动速度 (mm/s)
        """
        code, pose = self.arm.get_position()
        if code != 0:
            print("获取当前位置失败")
            return False

        current_x, current_y, current_z, roll, pitch, yaw = pose
        print(f"cur pos {pose}")

        # 计算目标位置
        target_pose = [
            current_x + dx,
            current_y + dy,
            current_z + dz,
            roll + dr,
            pitch + dp,
            yaw + dyaw
        ]

        # 执行直线运动
        code = self.arm.set_position(*target_pose, speed=speed, mvacc=1000, wait=True)
        if code == 0:
            print(f"🏃 移动完成: dx={dx}, dy={dy}, dz={dz}")
            return True
        else:
            print(f"⚠️ 移动失败，错误码: {code}")
            return False

    def set_gripper(self, position, speed=500):
        """
        控制 G2 夹爪开合
        :param position: 目标位置 (mm)。通常范围 0-850 (具体视型号而定，G2一般最大行程约85mm即850单位，或者0-100百分比，需根据SDK版本确认)
                         *注意：xarm-python-sdk 中 G2 默认单位通常是 mm*
        :param speed: 开合速度 (mm/s)
        """
        # G2 夹爪通常使用 set_gripper_position
        # 如果使用的是较新版本的 SDK，可能直接支持 mm
        code = self.arm.set_gripper_position(position, speed=speed, wait=True)

        if code == 0:
            print(f"🤏 夹爪动作完成: 目标位置 {position} mm")
            return True
        else:
            print(f"⚠️ 夹爪动作失败，错误码: {code}")
            return False

    def close_gripper(self, speed=500):
        """快捷方法：关闭夹爪 (假设完全闭合为 0mm)"""
        return self.set_gripper(0, speed)

    def open_gripper(self, max_open=850, speed=500):
        """快捷方法：打开夹爪 (假设最大张开为 850，即85mm，请根据实际物理极限调整)"""
        return self.set_gripper(max_open, speed)

    def disconnect(self):
        """断开连接"""
        self.arm.disconnect()
        print("👋 已断开连接")


# ================= 使用示例 =================
if __name__ == "__main__":
    # 1. 初始化
    robot = XArmController(ip="192.168.1.206")

    try:
        # 2. 移动测试
        print("--- 开始移动测试 ---")
        robot.move_relative(dz=50)   # 向上 50mm
        time.sleep(1)

        robot.move_relative(dx=50)   # 向右(X) 50mm
        time.sleep(1)

        robot.move_relative(dy=-50)  # 向Y轴负方向 50mm
        time.sleep(1)

        # 3. 夹爪测试
        print("--- 开始夹爪测试 ---")
        robot.open_gripper(max_open=800) # 张开到 80mm
        time.sleep(1)

        robot.close_gripper()          # 闭合
        time.sleep(1)

        # 4. 回到原位 (反向移动)
        robot.move_relative(dx=-50, dy=50, dz=-50)

    except KeyboardInterrupt:
        print("\n用户中断操作")
    finally:
        robot.disconnect()