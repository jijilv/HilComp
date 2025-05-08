import cv2
import numpy as np

def compare_and_save_images(image1_path, image2_path, output_path):
    # 读取图片
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 确保两张图片尺寸相同
    if image1.shape != image2.shape:
        print("两张图片尺寸不同，请确保它们尺寸相同。")
        return

    # 将图片转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 计算两个灰度图的差异
    difference = cv2.absdiff(gray1, gray2)

    # 将差异图二值化
    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # 找到差异区域的轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原图上用红色标记不同的部分
    for contour in contours:
        cv2.drawContours(image1, [contour], -1, (0, 0, 255), 1)

    # 保存结果图片
    cv2.imwrite(output_path, image1)
    print(f"输出图片已保存到 {output_path}")


# 使用示例
compare_and_save_images('/home/kemove/图片/truck-a.png', '/home/kemove/图片/truck-b.png','output_image.jpg')
