import os
import sys
from yolo_to_depth import YOLODepthExtractor
from depth_estimator import DepthEstimator

def run_demo(annotation_path, image_path, depth_path, calibration_idx=0, calibration_distance=50.0):
    """
    运行YOLO标注与深度图结合的完整演示
    
    Args:
        annotation_path: YOLO标注文件路径
        image_path: 图像文件路径
        depth_path: 深度图文件路径
        calibration_idx: 用于标定的标注索引（整数）或类别ID（字符串，如"0"或"class_0"）
        calibration_distance: 标定区域的实际距离（米）
    """
    print("===== YOLO标注与深度距离估计演示 =====")
    
    # 初始化提取器
    extractor = YOLODepthExtractor()
    
    # 1. 读取YOLO标注文件
    print(f"\n1. 读取标注文件: {annotation_path}")
    annotations = extractor.read_yolo_annotation(annotation_path)
    if not annotations:
        print("错误: 未读取到有效的标注")
        return
    print(f"   成功读取 {len(annotations)} 个标注物体")
    extractor.annotations = annotations
    
    # 2. 加载图像并获取尺寸
    print(f"\n2. 加载图像: {image_path}")
    image, width, height = extractor.load_image_info(image_path)
    if image is None:
        print("错误: 无法加载图像")
        return
    print(f"   图像尺寸: {width}x{height}")
    
    # 3. 将归一化坐标转换为像素坐标
    print("\n3. 转换标注坐标")
    pixel_boxes = extractor.convert_to_pixel_coordinates(annotations, width, height)
    extractor.pixel_boxes = pixel_boxes
    print(f"   成功转换 {len(pixel_boxes)} 个标注框坐标")
    
    # 4. 加载深度图
    print(f"\n4. 加载深度图: {depth_path}")
    depth_values = extractor.load_depth_map(depth_path)
    if depth_values is None:
        print("错误: 无法加载深度图")
        return
    
    # 5. 计算并输出深度图的深度值范围
    print("\n5. 深度图统计信息:")
    # 确保depth_values是numpy数组
    import numpy as np
    if isinstance(depth_values, np.ndarray):
        depth_min = float(np.min(depth_values))
        depth_max = float(np.max(depth_values))
        depth_mean = float(np.mean(depth_values))
        depth_std = float(np.std(depth_values))
        
        print(f"   深度图尺寸: {depth_values.shape}")
        print(f"   深度值范围: [{depth_min:.6f}, {depth_max:.6f}]")
        print(f"   平均深度值: {depth_mean:.6f}")
        print(f"   深度值标准差: {depth_std:.6f}")
    else:
        print("   警告: 无法计算非numpy数组类型的深度图统计信息")
    
    # 6. 提取标注区域的深度值
    print("\n6. 提取标注区域深度值")
    annotation_depths = extractor.extract_depth_from_annotations(depth_values)
    
    # 打印深度提取结果
    for idx, info in annotation_depths.items():
        print(f"   标注 {idx} (类别 {info['class_id']}):")
        print(f"     平均深度: {info['mean_depth']:.4f}")
        print(f"     深度范围: [{info['min_depth']:.4f}, {info['max_depth']:.4f}]")
    
    # 7. 使用指定区域进行距离标定
    print(f"\n7. 使用标注 {calibration_idx} 进行距离标定 (实际距离: {calibration_distance} 米)")
    scale_factor = extractor.calibrate_depth(calibration_idx, calibration_distance)
    if scale_factor is None:
        print("错误: 标定失败")
        return
    
    # 8. 计算所有区域的估计距离
    print("\n8. 计算所有标注区域的估计距离")
    extractor.estimate_all_distances()
    
    # 9. 显示所有区域的估计距离
    print("\n9. 所有标注区域的估计距离:")
    distance_count = 0
    for idx, info in extractor.annotation_depths.items():
        distance = info.get('estimated_distance', 'N/A')
        if distance != 'N/A':
            print(f"   标注 {idx} (类别 {info['class_id']}): 估计距离 = {distance:.2f} 米")
            distance_count += 1
        else:
            print(f"   标注 {idx} (类别 {info['class_id']}): 无法估计距离")
    
    # 10. 检查距离估计数量（用于后续相关系数计算）
    print("\n10. 距离估计总结:")
    print(f"   成功估计距离的物体数量: {distance_count}")
    print(f"   注意: 相关系数将在可视化时计算并显示（需要至少2个距离估计）")
    
    # 11. 保存距离估计结果
    results_dir = "depthanything_yolo_depth_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_file = os.path.join(results_dir, "distance_results.json")
    print(f"\n11. 保存距离估计结果到: {results_file}")
    extractor.save_distance_results(results_file)
    
    # 12. 可视化结果
    print("\n12. 生成可视化结果:")
    
    # 深度值可视化
    depth_viz_file = os.path.join(results_dir, "depth_visualization.jpg")
    extractor.visualize_annotations_with_depth(image, annotation_depths, depth_viz_file, show_distance=False)
    print(f"   深度值可视化已保存到: {depth_viz_file}")
    
    # 距离估计可视化（包含深度与距离相关系数显示）
    distance_viz_file = os.path.join(results_dir, "distance_visualization.jpg")
    extractor.visualize_annotations_with_depth(image, annotation_depths, distance_viz_file, show_distance=True, show_correlation=True)
    print(f"   距离估计可视化已保存到: {distance_viz_file}（包含深度与距离相关系数）")
    
    print("\n===== 演示完成 =====")
    print(f"所有结果文件保存在: {os.path.abspath(results_dir)}")


def main():
    # 直接使用硬编码的文件路径，无需命令行参数或实例
    print("使用硬编码文件路径运行深度距离估计...")
    
    # 以下路径请根据实际情况修改
    # annotation_path = r"C:/Users/11137/Desktop/100480000003090052_20250703125806.txt"  # YOLO标注文件
    # image_path = r"C:/Users/11137/Desktop/100480000003090052_20250703125806.jpg"  # 对应的图像文件

    annotation_path = r"/Users/oldshen/Desktop/20251130224730.txt"  # YOLO标注文件
    image_path = r"/Users/oldshen/Desktop/cbc779573478a4d17c36671921d5f931.jpg"  # 对应的图像文件

    model_path = r"/Users/oldshen/PycharmProjects/deepanything/deepview/Depth-Anything-V2-main/Depth-Anything-V2-main/Depth-Anything-V2-main/checkpoints/depth_anything_v2_vits.pth"
    save_dir = r"/Users/oldshen/PycharmProjects/deepanything/deepview/Depth-Anything-V2-main/Depth-Anything-V2-main/Depth-Anything-V2-main/depth_output"

    # 修复：encoder参数需要与模型文件匹配，vits改为vitl以匹配depth_anything_v2_vitl.pth
    estimator = DepthEstimator(model_path=model_path, encoder='vits')
     # 推理深度图 —— 得到 depth 数组 和 npy 文件路径
    depth, depth_npy_path = estimator.predict_depth(image_path=image_path, save_dir=save_dir)
    depth_path = depth_npy_path  # 深度图文件
    # 标定参数
    calibration_idx = "1"  # 使用类别0的第一个标注进行标定（可以是字符串类别ID或整数索引）
    calibration_distance = 500 # 假设该物体距离摄像头50米
    
    # 显示使用的参数
    print(f"\n使用参数:")
    print(f"- 标注文件: {annotation_path}")
    print(f"- 图像文件: {image_path}")
    print(f"- 深度图文件: {depth_path}")
    print(f"- 标定区域: {'类别ID' if isinstance(calibration_idx, str) else '索引'} {calibration_idx}")
    print(f"- 标定距离: {calibration_distance} 米")
    
    # 检查文件是否存在
    if not os.path.exists(annotation_path):
        print(f"错误：标注文件 '{annotation_path}' 不存在！")
        return
    
    if not os.path.exists(image_path):
        print(f"错误：图像文件 '{image_path}' 不存在！")
        return
    
    if not os.path.exists(depth_path):
        print(f"错误：深度图文件 '{depth_path}' 不存在！")
        print("请先使用task.py生成深度图：python task.py --img-path 你的图像路径 --outdir depth_output --save-npy")
        return
    
    # 运行演示
    run_demo(annotation_path, image_path, depth_path, calibration_idx, calibration_distance)


if __name__ == "__main__":
    main()