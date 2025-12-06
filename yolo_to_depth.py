import numpy as np
import os
import cv2
from typing import List, Dict, Tuple, Union

class YOLODepthExtractor:
    def __init__(self):
        """初始化YOLO深度提取器"""
        self.annotations = []
        self.pixel_boxes = []
        self.depth_values = None
        self.annotation_depths = {}
        self.scale_factor = None
        self.calibration_info = None
    
    def read_yolo_annotation(self, annotation_path):
        """
        读取YOLO格式的标注文件
        
        Args:
            annotation_path: YOLO标注文件路径
            
        Returns:
            annotations: 列表，每个元素为[class_id, center_x, center_y, width, height]
        """
        annotations = []
        
        try:
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1])  # 归一化坐标
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            annotations.append([class_id, center_x, center_y, width, height])
            return annotations
        except Exception as e:
            print(f"读取标注文件出错: {e}")
            return []
    
    def convert_to_pixel_coordinates(self, annotations, image_width, image_height):
        """
        将归一化坐标转换为像素坐标
        
        Args:
            annotations: YOLO标注列表
            image_width: 图像宽度
            image_height: 图像高度
            
        Returns:
            pixel_boxes: 列表，每个元素为[class_id, x_min, y_min, x_max, y_max]
        """
        pixel_boxes = []
        
        for ann in annotations:
            class_id, center_x, center_y, width, height = ann
            
            # 转换为像素坐标
            center_x_pixel = int(center_x * image_width)
            center_y_pixel = int(center_y * image_height)
            width_pixel = int(width * image_width)
            height_pixel = int(height * image_height)
            
            # 计算边界框坐标
            x_min = max(0, center_x_pixel - width_pixel // 2)
            y_min = max(0, center_y_pixel - height_pixel // 2)
            x_max = min(image_width - 1, center_x_pixel + width_pixel // 2)
            y_max = min(image_height - 1, center_y_pixel + height_pixel // 2)
            
            pixel_boxes.append([class_id, x_min, y_min, x_max, y_max])
        
        return pixel_boxes
    
    def load_image_info(self, image_path):
        """
        加载图像并获取尺寸信息
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            tuple: (image, width, height) 或 (None, 0, 0)
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None, 0, 0
            
            height, width = image.shape[:2]
            return image, width, height
        except Exception as e:
            print(f"加载图像出错: {e}")
            return None, 0, 0
    
    def load_depth_map(self, depth_path):
        """
        加载深度图数据
        
        Args:
            depth_path: 深度图文件路径 (支持.npy格式)
            
        Returns:
            depth_values: numpy数组的深度值
        """
        try:
            if depth_path.endswith('.npy'):
                depth_values = np.load(depth_path)
                print("从NPY文件加载深度图成功！")
            elif depth_path.endswith(('.png', '.jpg', '.jpeg')):
                # 从图像加载深度值（假设为灰度图）
                depth_values = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            else:
                print(f"不支持的深度图格式: {depth_path}")
                return None
            
            self.depth_values = depth_values
            print(f"深度图加载成功，形状: {depth_values.shape}")
            return depth_values
        except Exception as e:
            print(f"加载深度图出错: {e}")
            return None
    
    def extract_depth_from_annotations(self, depth_values=None):
        """
        从标注区域提取深度值
        
        Args:
            depth_values: 深度值数组，如果为None则使用已加载的值
            
        Returns:
            annotation_depths: 字典，键为标注索引，值为区域深度统计信息
        """
        if not self.pixel_boxes:
            print("请先加载并解析标注文件")
            return {}
        
        if depth_values is None:
            depth_values = self.depth_values
        
        if depth_values is None:
            print("请先加载深度图")
            return {}
        
        annotation_depths = {}
        
        for idx, box in enumerate(self.pixel_boxes):
            class_id, x_min, y_min, x_max, y_max = box
            
            # 确保坐标在深度图范围内
            depth_height, depth_width = depth_values.shape
            x_min = max(0, min(x_min, depth_width - 1))
            y_min = max(0, min(y_min, depth_height - 1))
            x_max = max(0, min(x_max, depth_width - 1))
            y_max = max(0, min(y_max, depth_height - 1))
            
            # 提取区域深度值
            region_depth = depth_values[y_min:y_max+1, x_min:x_max+1]
            
            if region_depth.size > 0:
                # 计算统计信息
                stats = {
                    'class_id': class_id,
                    'box': box,
                    'mean_depth': float(np.mean(region_depth)),
                    'median_depth': float(np.median(region_depth)),
                    'min_depth': float(np.min(region_depth)),
                    'max_depth': float(np.max(region_depth)),
                    'std_depth': float(np.std(region_depth)),
                    'area': region_depth.size,
                    'estimated_distance': None  # 将在标定时计算
                }
                annotation_depths[idx] = stats
            else:
                print(f"标注 {idx} 的区域为空")
        
        self.annotation_depths = annotation_depths
        return annotation_depths
    
    def calibrate_depth(self, annotation_idx, actual_distance, use_mean=True):
        """
        使用指定标注区域的实际距离进行深度标定
        
        Args:
            annotation_idx: 标注索引或类别ID（如果是整数则作为索引，否则作为类别ID字符串）
            actual_distance: 该区域的实际距离（米）
            use_mean: 是否使用平均深度值进行标定
            
        Returns:
            scale_factor: 计算得到的比例因子
        """
        if not self.annotation_depths:
            print("请先提取标注区域的深度值")
            return None
        
        # 检查annotation_idx是否为类别ID字符串（如"class_0"或"0"）
        target_idx = None
        if isinstance(annotation_idx, str):
            # 尝试将字符串解析为类别ID
            if annotation_idx.startswith("class_"):
                class_id_str = annotation_idx[6:]
            else:
                class_id_str = annotation_idx
            
            try:
                target_class_id = int(class_id_str)
                # 查找对应类别的第一个标注
                for idx, depth_info in self.annotation_depths.items():
                    if depth_info['class_id'] == target_class_id:
                        target_idx = idx
                        print(f"找到类别 {target_class_id} 的标注，使用索引 {target_idx}")
                        break
                
                if target_idx is None:
                    print(f"未找到类别 {target_class_id} 的标注")
                    return None
            except ValueError:
                print(f"无效的类别ID格式: {annotation_idx}")
                return None
        else:
            # 使用索引方式
            target_idx = annotation_idx
            if target_idx not in self.annotation_depths:
                print(f"标注索引 {target_idx} 不存在")
                return None
        
        # 获取该区域的深度值
        depth_info = self.annotation_depths[target_idx]
        depth_value = depth_info['mean_depth'] if use_mean else depth_info['median_depth']
        
        # 计算比例因子（相对深度与实际距离成反比）
        # 实际距离 = 比例因子 / 相对深度值
        scale_factor = actual_distance * depth_value
        
        self.scale_factor = scale_factor
        self.calibration_info = {
            'annotation_idx': target_idx,
            'original_idx': annotation_idx,
            'class_id': depth_info['class_id'],
            'actual_distance': actual_distance,
            'depth_value_used': depth_value,
            'scale_factor': scale_factor,
            'use_mean': use_mean
        }
        
        print(f"深度标定完成:")
        print(f"  标定区域: 索引 {target_idx}, 类别 {depth_info['class_id']}")
        if isinstance(annotation_idx, str):
            print(f"  使用类别ID: {annotation_idx} 进行选择")
        print(f"  实际距离: {actual_distance} 米")
        print(f"  使用的深度值: {depth_value}")
        print(f"  计算的比例因子: {scale_factor}")
        
        # 更新所有标注区域的估计距离
        self.estimate_all_distances()
        
        return scale_factor
    
    def estimate_all_distances(self):
        """
        使用标定的比例因子估计所有标注区域的实际距离
        
        Returns:
            updated_annotations: 更新后的标注信息（包含估计距离）
        """
        if self.scale_factor is None:
            print("请先进行深度标定")
            return self.annotation_depths
        
        for idx, depth_info in self.annotation_depths.items():
            # 使用与标定时相同的深度值类型
            if self.calibration_info['use_mean']:
                depth_value = depth_info['mean_depth']
            else:
                depth_value = depth_info['median_depth']
            
            # 避免除零错误
            if depth_value > 0:
                # 实际距离 = 比例因子 / 相对深度值
                estimated_distance = self.scale_factor / depth_value
                depth_info['estimated_distance'] = estimated_distance
            else:
                depth_info['estimated_distance'] = float('inf')
                print(f"警告: 标注 {idx} 的深度值为0，无法估计距离")
        
        return self.annotation_depths
    
    def get_distance_estimation_results(self):
        """
        获取所有标注区域的距离估计结果
        
        Returns:
            results: 字典，包含距离估计结果和标定信息
        """
        results = {
            'calibration_info': self.calibration_info,
            'scale_factor': self.scale_factor,
            'annotations': []
        }
        
        for idx, depth_info in self.annotation_depths.items():
            annotation_result = {
                'index': idx,
                'class_id': depth_info['class_id'],
                'box': depth_info['box'],
                'mean_depth': depth_info['mean_depth'],
                'estimated_distance': depth_info.get('estimated_distance', None)
            }
            results['annotations'].append(annotation_result)
        
        return results
    
    def save_distance_results(self, output_path):
        """
        保存距离估计结果到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        import json
        
        results = self.get_distance_estimation_results()
        
        # 将numpy类型转换为Python原生类型以便JSON序列化
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # 递归转换所有numpy类型
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_numpy_types(data)
        
        converted_results = recursive_convert(results)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, ensure_ascii=False, indent=2)
            print(f"距离估计结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存结果出错: {e}")
    
    def calculate_depth_distance_correlation(self, annotation_depths=None):
        """
        计算深度值与距离估计值之间的相关系数
        
        Args:
            annotation_depths: 标注区域的深度信息（如果为None则使用已存储的）
            
        Returns:
            correlation: 相关系数值，接近-1表示深度值与距离成反比关系良好
        """
        if annotation_depths is None:
            annotation_depths = self.annotation_depths
        
        # 提取有距离估计的深度值和距离值
        depths = []
        distances = []
        
        for stats in annotation_depths.values():
            if 'estimated_distance' in stats and stats['estimated_distance'] is not None:
                depths.append(stats['mean_depth'])
                distances.append(stats['estimated_distance'])
        
        # 只有当有足够的数据点时才计算相关系数
        if len(depths) >= 2:
            # 计算相关系数
            correlation = np.corrcoef(depths, distances)[0, 1]
            return correlation
        return None
    
    def visualize_annotations_with_depth(self, image, annotation_depths=None, output_path=None, show_distance=False, show_correlation=False):
        """
        可视化标注区域和深度/距离信息
        
        Args:
            image: 原始图像
            annotation_depths: 标注区域的深度信息（如果为None则使用已存储的）
            output_path: 输出图像保存路径
            show_distance: 是否显示估计距离（如果已标定）
            show_correlation: 是否显示深度与距离的相关系数
            
        Returns:
            visualized_image: 可视化后的图像
        """
        if image is None:
            print("无法可视化，图像为None")
            return None
        
        # 使用提供的深度信息或已存储的
        if annotation_depths is None:
            annotation_depths = self.annotation_depths
        
        if not annotation_depths:
            print("没有标注深度信息可可视化")
            return None
        
        # 创建图像副本
        visualized = image.copy()
        
        # 为标定区域使用不同颜色
        calibration_idx = self.calibration_info['annotation_idx'] if self.calibration_info else -1
        
        for idx, stats in annotation_depths.items():
            class_id = stats['class_id']
            x_min, y_min, x_max, y_max = stats['box'][1:]
            mean_depth = stats['mean_depth']
            
            # 为标定区域使用红色，其他区域使用绿色
            color = (0, 0, 255) if idx == calibration_idx else (0, 255, 0)
            
            # 绘制边界框
            cv2.rectangle(visualized, (x_min, y_min), (x_max, y_max), color, 2)
            
            # 显示信息
            if show_distance and 'estimated_distance' in stats and stats['estimated_distance'] is not None:
                # 显示估计距离
                distance = stats['estimated_distance']
                text = f"Class {class_id}: {distance:.2f}m"
                # 对于标定区域，添加特殊标记
                if idx == calibration_idx:
                    text += " (Calibrated)"  # 使用英文替代中文
            else:
                # 显示相对深度值
                text = f"Class {class_id}: Depth={mean_depth:.2f}"
            
            cv2.putText(visualized, text, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 如果已标定，在图像上显示标定信息
        if show_distance and self.calibration_info:
            calib_text = f"Calibration: Scale={self.scale_factor:.2f}"  # 使用英文替代中文
            cv2.putText(visualized, calib_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 计算深度与距离的相关系数
        if show_correlation and show_distance:
            correlation = self.calculate_depth_distance_correlation(annotation_depths)
            
        # 在每个目标的边界框下方显示相关系数
        for idx, stats in annotation_depths.items():
            x_min, y_min, x_max, y_max = stats['box'][1:]
            # 为标定区域使用红色，其他区域使用绿色
            calibration_idx = self.calibration_info['annotation_idx'] if self.calibration_info else -1
            color = (0, 0, 255) if idx == calibration_idx else (0, 255, 0)
            
            # 显示相关系数
            if show_correlation and show_distance and correlation is not None:
                # 根据相关系数选择显示颜色
                if correlation < -0.9:
                    corr_color = (0, 255, 0)  # 绿色 - 良好
                elif correlation < -0.7:
                    corr_color = (0, 255, 255)  # 黄色 - 可接受
                else:
                    corr_color = (0, 0, 255)  # 红色 - 不佳
                
                corr_text = f"Corr: {correlation:.3f}"
                # 显示在边界框下方
                cv2.putText(visualized, corr_text, (x_min, y_max + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, corr_color, 2)
        
        # 保存可视化结果
        if output_path:
            cv2.imwrite(output_path, visualized)
            print(f"可视化结果已保存到: {output_path}")
        
        return visualized

# 示例使用
def main():
    extractor = YOLODepthExtractor()
    
    # 这里仅作为示例框架，实际使用时需要根据具体文件路径修改
    print("YOLO深度提取器初始化成功")
    print("使用流程:")
    print("1. 调用read_yolo_annotation读取标注文件")
    print("2. 调用load_image_info获取图像尺寸")
    print("3. 调用convert_to_pixel_coordinates转换坐标")
    print("4. 调用load_depth_map加载深度图")
    print("5. 调用extract_depth_from_annotations提取标注区域深度值")
    print("6. 调用visualize_annotations_with_depth可视化结果")
    
    # 示例代码框架
    # annotation_path = "path/to/annotation.txt"
    # image_path = "path/to/image.jpg"
    # depth_path = "path/to/depth.npy"
    # 
    # # 读取标注
    # annotations = extractor.read_yolo_annotation(annotation_path)
    # extractor.annotations = annotations
    # 
    # # 获取图像信息并转换坐标
    # image, width, height = extractor.load_image_info(image_path)
    # pixel_boxes = extractor.convert_to_pixel_coordinates(annotations, width, height)
    # extractor.pixel_boxes = pixel_boxes
    # 
    # # 加载深度图并提取深度值
    # depth_values = extractor.load_depth_map(depth_path)
    # annotation_depths = extractor.extract_depth_from_annotations(depth_values)
    # 
    # # 可视化结果
    # if image is not None:
    #     output_path = "visualization_with_depth.jpg"
    #     extractor.visualize_annotations_with_depth(image, annotation_depths, output_path)

if __name__ == "__main__":
    main()