# Combine-Is_Bot-with-anygrasp

# 🤖 AnyGrasp 与 Kinova Gen3 集成项目提示词

---

## 📋 **项目背景**

我们有一个名为 **IS_Bot** 的机器人控制项目，用于驱动 **Kinova Gen3 机械臂**。现在需要集成 **AnyGrasp** 视觉抓取检测模型，实现**完全自动化的物体抓取**功能。

AnyGrasp SDK 已部署在 `~/documents/anygrasp_sdk`。

---

## 🎯 **集成目标**

实现一个完整的自动抓取管道：
```
RGB-D 相机图像 → AnyGrasp 抓取检测 → 坐标转换 → 逆运动学求解 
→ 轨迹规划 → Kinova Gen3 执行
```

---

## 📁 **现有项目结构**

**关键文件和模块：**

- main.py - 主程序入口，控制循环
- real_env.py - 真实环境接口（机器人通信、传感器读取）
- mujoco_env.py - 仿真环境
- arm_server.py - 机械臂 RPC 服务器
- ik_solver.py - **逆运动学求解器**（Jacobian 方法）
- arm_controller.py - 低层力矩控制
- kinova.py - Kinova SDK 封装
- policies.py - 控制策略基类（`Policy`）
- constants.py - 项目常量配置
- episode_storage.py - 数据存储模块

**关键类的接口：**

```python
# Policy 基类
class Policy:
    def reset(self):
        """重置策略状态"""
        raise NotImplementedError
    
    def step(self, obs) -> dict:
        """
        输入观测，返回动作字典
        Returns: {"arm_pos": [x,y,z], "arm_quat": [qx,qy,qz,qw], "gripper_pos": [0-1]}
        或 "end_episode"/"reset_env"
        """
        raise NotImplementedError

# RealEnv 环境类
class RealEnv:
    def get_obs(self) -> dict:
        """返回 {"arm_pos": [...], "arm_quat": [...], "gripper_pos": [...]}"""
    
    def step(self, action: dict):
        """执行动作，action 格式同上"""
    
    def reset(self):
        """重置环境"""
    
    def close(self):
        """关闭连接"""
```

---

## 🛠️ **需要实现的模块**

### **1. AnyGraspWrapper（`policies/anygrasp_wrapper.py`）**

**功能：** 封装 AnyGrasp 推理接口

```python
class AnyGraspWrapper:
    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        初始化 AnyGrasp 模型
        
        Args:
            model_path: 模型权重路径（如果为 None，使用默认路径）
            use_gpu: 是否使用 GPU
        """
        
    def predict(self, rgb: np.ndarray, depth: np.ndarray) -> list:
        """
        执行抓取检测推理
        
        Args:
            rgb: RGB 图像 [H, W, 3]，uint8
            depth: 深度图 [H, W]，单位：米（float32）
        
        Returns:
            grasp_candidates: 抓取候选列表，每个元素包含：
            {
                'position': np.array([x, y, z]),      # 3D 抓取点（相机坐标系）
                'approach_direction': np.array([...]), # 接近方向（3 维单位向量）
                'angle': float,                         # 旋转角度
                'width': float,                         # 夹爪宽度
                'score': float,                         # 抓取质量评分 [0, 1]
            }
        
        按 score 降序排列
        """
```

---

### **2. 坐标系转换器（`robot_controller/grasp_converter.py`）**

**功能：** 将相机坐标系的抓取点转换为机器人末端位姿

```python
class GraspConverter:
    def __init__(self, camera_intrinsics: dict = None, 
                 camera_to_base_transform: np.ndarray = None):
        """
        初始化坐标转换器
        
        Args:
            camera_intrinsics: 相机内参 {"fx", "fy", "cx", "cy"}
            camera_to_base_transform: 4x4 变换矩阵（相机→机器人基坐标系）
        """
    
    def grasp_to_ee_pose(self, grasp: dict, 
                         approach_distance: float = 0.05) -> tuple:
        """
        将抓取候选转换为末端执行器目标位姿
        
        Args:
            grasp: AnyGrasp 输出的抓取字典
            approach_distance: 接近距离（在接近方向上的偏移）
        
        Returns:
            (position, quaternion): 
            - position: np.array([x, y, z])，机器人基坐标系
            - quaternion: np.array([qx, qy, qz, qw])
        
        需要考虑：
        1. 夹爪的几何约束（宽度、长度）
        2. 接近方向与末端执行器方向的对应关系
        3. 安全的接近轨迹
        """
    
    def verify_reachability(self, position: np.ndarray, 
                           quaternion: np.ndarray, 
                           ik_solver) -> bool:
        """
        验证目标位姿是否可达
        
        Args:
            position, quaternion: 目标末端位姿
            ik_solver: IKSolver 实例
        
        Returns:
            是否可达（IK 有解）
        """
```

---

### **3. 自动抓取策略（`policies/grasp_policy.py`）**

**功能：** 实现完整的自动抓取控制策略

```python
class GraspPolicy(Policy):
    def __init__(self, anygrasp_model_path: str = None,
                 camera_to_base_transform: np.ndarray = None,
                 ik_solver = None,
                 max_attempts: int = 5,
                 min_grasp_score: float = 0.5):
        """
        初始化自动抓取策略
        
        Args:
            anygrasp_model_path: AnyGrasp 模型路径
            camera_to_base_transform: 相机→机器人坐标系变换矩阵
            ik_solver: IKSolver 实例（可选，如果为 None 内部创建）
            max_attempts: 最多尝试的抓取候选数
            min_grasp_score: 最小可接受的抓取评分
        """
    
    def reset(self):
        """重置策略状态"""
        
    def step(self, obs: dict):
        """
        执行一步自动抓取
        
        Args:
            obs: 环境观测 {"arm_pos": [...], "arm_quat": [...], 
                          "gripper_pos": [...], 
                          可选: "wrist_rgb": [...], "wrist_depth": [...]}
        
        Returns:
            动作字典或控制命令：
            - 首次调用：返回 None（等待用户确认开始）
            - 执行中：返回动作字典 {"arm_pos": [...], ...}
            - 任务完成：返回 "end_episode"
        
        流程：
        1. 从观测中提取 RGB-D 图像
        2. 调用 AnyGrasp 检测抓取点
        3. 按评分筛选候选（score >= min_grasp_score）
        4. 遍历候选，尝试逆运动学求解
        5. 第一个可达的抓取：生成接近→抓取→回收的动作序列
        6. 返回动作，进入下一状态
        """
    
    def _grasp_to_action_sequence(self, grasp: dict, 
                                  current_pose: tuple) -> list:
        """
        将单个抓取候选转换为动作序列（接近→抓取→回收）
        
        Returns:
            动作列表，每个元素是标准动作字典
        """
```

---

## 📊 **集成流程**

### **使用方式：**

```bash
# 启动机械臂 RPC 服务
python arm_server.py

# 运行自动抓取（真实环境）
python main.py --grasp

# 运行自动抓取（仿真环境，可视化）
python main.py --sim --grasp --sim-showing
```

### **修改 main.py：**

需要在 `main()` 函数中添加对 `GraspPolicy` 的支持：

```python
def main(args):
    # ... 现有代码 ...
    
    if args.teleop:
        policy = TeleopPolicy()
    elif args.grasp:  # 新增
        policy = GraspPolicy(ik_solver=env.arm.ik_solver)  # 需要从环境获取
    else:
        policy = RemotePolicy()
    
    # ... 其他代码 ...
```

---

## 🔧 **技术要点和约束**

### **相机标定：**
- 需要获取或标定相机内参和外参（相机→机器人基坐标系）
- 可以使用 RealSense 内参或 OpenCV 标定工具

### **坐标系定义：**
- AnyGrasp 输出：相机坐标系（通常 Z 向前，X 向右）
- 机器人坐标系：Kinova Gen3 基坐标系
- 需要明确 RGB-D 图像方向（可能需要旋转/翻转）

### **抓取执行策略：**
1. **接近阶段**：从当前位置移动到接近位置（接近方向反向）
2. **抓取阶段**：打开夹爪→接近抓取点→闭合夹爪
3. **回收阶段**：抬起物体→返回安全位置

### **安全考虑：**
- 关节极限检查
- 自碰撞检测（可选）
- 力反馈监控（物体滑脱检测）

---

## 📦 **依赖项**

需要的包已在 requirements.txt 中，额外需要：
- AnyGrasp SDK（已部署在 `~/documents/anygrasp_sdk`）
- OpenCV（用于图像处理）

---

## ✅ **验收标准**

1. **AnyGraspWrapper** 能正确加载模型并进行推理
2. **GraspConverter** 能准确转换坐标系
3. **GraspPolicy** 能生成有效的动作序列
4. 在仿真环境中成功执行完整的抓取循环
5. 代码有清晰的注释和错误处理

---

## 💡 **实现提示**

- 优先在仿真环境（MuJoCo）中测试和调试
- 使用 mujoco_env.py 中的可视化功能来验证坐标变换
- 可参考 policies.py 中 `TeleopPolicy` 的实现方式
- 数据流向：观测 → 策略 → 动作 → 环境执行

---

这是一个用于指导另一个 AI 代理实现 AnyGrasp 集成的完整提示词。你可以直接将其发给 Claude Sonnet 4.5。
