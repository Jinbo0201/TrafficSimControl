import time
import numpy as np
import matplotlib.pyplot as plt
import random
import traceback

from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import StraightLane, LineType, SineLane # 新增: 导入 SineLane 以支持曲线汇入
from highway_env.vehicle.behavior import IDMVehicle

def test_merge_env():
    """
    构建一个包含主路和匝道的 Merge 路网结构测试程序。
    修复了 add_lane 参数错误，并完全参照 merge_env.py 的路网构建逻辑。
    """

    # 1) 构建 Merge 路网 (参照 merge_env.py)
    net = RoadNetwork()
    
    # 定义路段长度比例 (参照 merge_env: Before, converging, merge, after)
    ends = [150, 80, 80, 150] 
    c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
    
    try:
        # --- 主路车道 (Highway lanes) ---
        # 设置两条主路车道的高度
        y = [0, StraightLane.DEFAULT_WIDTH]
        # 定义车道线类型
        line_type = [[c, s], [n, c]]
        line_type_merge = [[c, s], [n, s]]
        
        for i in range(2):
            # 路段 a -> b (汇入前)
            net.add_lane(
                "a",
                "b",
                StraightLane([0, y[i]], [sum(ends[:2]), y[i]], line_types=line_type[i]),
            )
            # 路段 b -> c (汇入区)
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [sum(ends[:2]), y[i]],
                    [sum(ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            # 路段 c -> d (汇入后)
            net.add_lane(
                "c",
                "d",
                StraightLane(
                    [sum(ends[:3]), y[i]], [sum(ends), y[i]], line_types=line_type[i]
                ),
            )

        # --- 匝道车道 (Merging lane) ---
        # 参照 merge_env 的几何构建方式
        amplitude = 3.25
        # 匝道起始段 (直线)
        ljk = StraightLane(
            [0, 6.5 + 4 + 4], [ends[0], 6.5 + 4 + 4], line_types=[c, c], forbidden=True
        )
        # 匝道汇入段 (正弦曲线，平滑连接到主路)
        lkb = SineLane(
            ljk.position(ends[0], -amplitude),
            ljk.position(sum(ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * ends[1]),
            np.pi / 2,
            line_types=[c, c],
            forbidden=True,
        )
        # 匝道合并后的直行段 (与主路 b->c 平行/重合逻辑，但在不同节点定义以区分车流)
        # 注意：这里为了简化测试，我们让匝道车在物理上汇入主路区域
        # 在 merge_env 中，lbc 是连接到 ("b", "c") 的另一个车道索引 (通常是 2)
        lbc = StraightLane(
            lkb.position(ends[1], 0),
            lkb.position(ends[1], 0) + [ends[2], 0],
            line_types=[n, c],
            forbidden=True,
        )
        
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc) # 这将创建 ("b", "c", 2) 车道

    except Exception as e:
        print(f"路网构建失败：{e}")
        traceback.print_exc()
        return

    road = Road(network=net, np_random=np.random.RandomState(0), record_history=False)
    
    # 可选：添加障碍物模拟汇入点结束（参照 merge_env，非必须但增加真实感）
    # from highway_env.vehicle.objects import Obstacle
    # road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))

    # 2) 定义加车辅助函数
    def add_car(lane_index, s, speed, target_speed=None):
        try:
            # lane_index 应该是 ("from", "to", id) 元组
            if len(lane_index) == 2:
                # 兼容旧调用，默认 id=0
                lane_index = (lane_index[0], lane_index[1], 0)
            
            lane = road.network.get_lane(lane_index)
            if lane is None:
                raise ValueError(f"车道不存在：{lane_index}")
                
            pos = lane.position(s, 0)
            heading = lane.heading_at(s)
            
            v = IDMVehicle(
                road,
                pos,
                heading=heading,
                speed=speed,
                target_speed=target_speed if target_speed is not None else speed,
                target_lane_index=lane_index,
            )
            road.vehicles.append(v)
            return v
        except Exception as e:
            print(f"[ERROR] 添加车辆失败 {lane_index}: {e}")
            traceback.print_exc()
            return None

    # 初始化车辆
    # 主车：在主路 ("a", "b", 1) 行驶 (参照 merge_env  ego_vehicle 位置)
    # 注意：merge_env 中 ego 在 ("a", "b", 1)，即右侧车道或左侧车道取决于定义，这里 y=[0, WIDTH], 1 是上方车道
    add_car(("a", "b", 1), 30.0, 30.0, 30.0)
    
    # 其他主路车辆
    other_positions = [(90.0, 29.0), (70.0, 31.0), (5.0, 31.5)]
    for pos, spd in other_positions:
        lane_idx = ("a", "b", 0) # 简单起见放在车道 0
        # 增加一点随机性
        import random
        pos += random.uniform(-5.0, 5.0)
        spd += random.uniform(-1.0, 1.0)
        add_car(lane_idx, pos, spd, spd)

    # 匝道车辆：在 ("j", "k", 0)
    merging_v = add_car(("j", "k", 0), 110.0, 20.0, 30.0)
    if merging_v:
        # 设置目标车道为主路汇入段 ("b", "c", 0) 或 ("b", "c", 1)，IDM 会尝试变道
        # 在 merge_env 中，汇入车辆通常目标是 ("b", "c", 0) 或类似，这里设为 ("b", "c", 0) 尝试汇入
        merging_v.target_lane_index = ("b", "c", 0)

    # 3) 可视化准备
    sim_fps = 15
    dt = 1.0 / sim_fps
    max_steps = 600

    fig, ax = plt.subplots(figsize=(12, 5))
    plt.ion()

    # 预计算路网线条用于绘制背景
    lanes_to_draw = []
    for _from, _to_dict in net.graph.items():
        for _to, _lanes_list in _to_dict.items():
            # 修改：_lanes_list 是列表而非字典，使用 enumerate 遍历
            for _id, lane in enumerate(_lanes_list):
                # 采样绘制曲线
                start = lane.position(0, 0)
                end = lane.position(lane.length, 0)
                # 对于曲线，多采几个点
                xs, ys = [], []
                for t in np.linspace(0, lane.length, 20):
                    p = lane.position(t, 0)
                    xs.append(p[0])
                    ys.append(p[1])
                lanes_to_draw.append((xs, ys))

    step_counter = 0
    for step in range(max_steps):
        try:
            road.act()
            road.step(dt)

            # 动态加车逻辑 (可选，暂时注释掉以避免过于拥挤，专注于初始场景调试)
            # if step % 40 == 20:
            #     ...

            # 清理已驶出路网的车辆
            valid_vehicles = []
            max_x = sum(ends) + 50 # 总长度 + 缓冲
            for v in road.vehicles:
                if np.any(np.isnan(v.position)) or np.any(np.isinf(v.position)):
                    continue
                if v.position[0] < max_x:
                    valid_vehicles.append(v)
            road.vehicles = valid_vehicles

            # 绘图更新
            ax.clear()
            ax.set_title(f"Merge Simulation (Fixed) - Step {step}    t={step*dt:.2f}s")
            # 动态调整视野
            all_x = [v.position[0] for v in road.vehicles] if road.vehicles else [0]
            center_x = np.mean(all_x) if all_x else 200
            view_range = 200
            ax.set_xlim(center_x - view_range/2, center_x + view_range/2)
            ax.set_ylim(-10, 15) # 根据 y 坐标范围设定
            ax.set_aspect('equal')

            # 绘制路网背景
            for xs, ys in lanes_to_draw:
                ax.plot(xs, ys, color='k', linewidth=2)
            
            # 绘制车辆
            for v in road.vehicles:
                if np.any(np.isnan(v.position)):
                    continue
                color = "red" if getattr(v, 'crashed', False) else "blue"
                ax.scatter(v.position[0], v.position[1], color=color, s=100, edgecolors='black')
                ax.text(v.position[0] + 2, v.position[1] + 0.5, f"{v.speed:.1f}", fontsize=9, color='darkred')
                
                dx = np.cos(v.heading) * 2
                dy = np.sin(v.heading) * 2
                ax.arrow(v.position[0], v.position[1], dx, dy, head_width=1.5, head_length=1, fc=color, ec=color)

            plt.pause(0.001)

            if any(getattr(v, 'crashed', False) for v in road.vehicles):
                print(f"碰撞发生 at step {step}")

        except Exception as e:
            print(f"[FATAL] 仿真循环在第 {step} 步崩溃：{e}")
            traceback.print_exc()
            break

    plt.ioff()
    plt.show()
    print("Merge 模拟结束")


if __name__ == "__main__":
    test_merge_env()