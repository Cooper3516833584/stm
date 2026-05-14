import time
import cv2
from loguru import logger
from FlightController.Components.LDRadar_Driver import LD_Radar
from FlightController.Solutions.Radar_SLAM import radar_resolve_rt_pose

def main():
    logger.info("=== 阶段 M3: 霍夫变换直线检测无头渲染测试 ===")
    radar = LD_Radar(name="D500_SLAM_Test")
    radar.start(com="/dev/ttySTM4", radar_type="D500")
    
    try:
        logger.info("等待 5 秒让雷达积累足够的环境点云建立 Map_Circle...")
        time.sleep(5)
        
        if radar.connected and radar.map.avail_points > 500:
            logger.info("正在从 Map_Circle 提取多边形点云图像矩阵...")
            
            # 使用源码自带的方法生成点云图像 (Size: 800x800, Scale=0.1)
            img_matrix = radar.map.output_polyline_cloud(
                scale=0.1, size=800, thickness=2, draw_outside=False
            )
            
            logger.info("启动霍夫直线位姿解析...")
            # 必须强制传入 debug=True 和 debug_save_img=True，绕开 imshow 陷阱！
            x, y, yaw = radar_resolve_rt_pose(
                img_matrix, 
                debug=True, 
                debug_save_img=True, 
                skip_di=False, 
                skip_er=False
            )
            
            logger.success(f"解析完成！提取到位姿: X={x}, Y={y}, YAW={yaw}")
            logger.info("请在当前目录下查看生成的 'radar_resolve_debug.png' 图像！")
        else:
            logger.error("点云数据不足或雷达未连接，无法执行算法测试。")

    except Exception as e:
        logger.exception("算法执行发生异常！")
    finally:
        radar.stop()

if __name__ == "__main__":
    main()