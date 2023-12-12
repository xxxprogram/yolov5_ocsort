import os
import xml.etree.ElementTree as ET


def convert_xml_to_txt(image_path, xml_path, output_folder, names):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图片尺寸
    image_width = float(root.find("size").find("width").text)
    image_height = float(root.find("size").find("height").text)

    # 创建输出TXT文件路径
    file_name = os.path.basename(image_path)
    txt_file_path = os.path.join(output_folder, file_name + ".txt")

    # 打开TXT文件以写入坐标信息
    with open(txt_file_path, "w") as f:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            cls_id = names.index(class_name)
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # 检查边界是否超出图像范围
            if xmin < 0 or ymin < 0 or xmax > image_width or ymax > image_height:
                print(f"警告：边界框超出图像范围！({xmin}, {ymin}, {xmax}, {ymax})")
                continue

            # 计算边界框的中心点坐标和宽高
            x = (xmin + xmax) / 2 / image_width
            y = (ymin + ymax) / 2 / image_height
            w = (xmax - xmin) / image_width
            h = (ymax - ymin) / image_height

            # 将坐标信息写入TXT文件
            f.write(f"{cls_id} {x} {y} {w} {h}\n")

    print(f"成功将XML文件转换为TXT文件：{txt_file_path}")


if __name__ == '__main__':
    # 设置输入图片路径、XML路径和输出文件夹路径
    img_path = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\Yolov5_OCtrack\datasets\airplane\img"
    xml_path = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\Yolov5_OCtrack\datasets\airplane\xml"
    output_folder = r"D:\lg\BaiduSyncdisk\project\person_code\project_self\Yolov5_OCtrack\datasets\airplane\txt"

    names = ['airplane']
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(xml_path):
        for f in files:
            filename, ext = os.path.splitext(f)
            # 判断是否为xml文件
            if ext != '.xml':
                continue
            # 判断图片是否存在
            image_path = os.path.join(img_path, filename)
            xmlpath = os.path.join(xml_path, f)
            # 调用函数进行转换
            convert_xml_to_txt(image_path, xmlpath, output_folder, names)
