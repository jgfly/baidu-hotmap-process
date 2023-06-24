from PIL import Image

import cv2
import numpy as np

def divide_blend(top_image, bottom_image, output_image):
    # 打开图像
    top = Image.open(top_image).convert("RGB")
    bottom = Image.open(bottom_image).convert("RGB")
    
    # 确保图像尺寸相同
    if top.size != bottom.size:
        top = top.resize(bottom.size)
    
    # 创建新的图像对象
    result = Image.new('RGB', top.size)
    
    # 获取像素数据
    top_pixels = top.load()
    bottom_pixels = bottom.load()
    result_pixels = result.load()
    
    # 应用划分混合模式
    for x in range(top.width):
        for y in range(top.height):
            # 获取各个通道的颜色值
            r_top, g_top, b_top = top_pixels[x, y]
            r_bottom, g_bottom, b_bottom = bottom_pixels[x, y]
            
            # 应用划分混合模式
            r_result = 255 if r_bottom == 0 else min(255, r_top * 255 // r_bottom)
            g_result = 255 if g_bottom == 0 else min(255, g_top * 255 // g_bottom)
            b_result = 255 if b_bottom == 0 else min(255, b_top * 255 // b_bottom)
            
            # 设置结果图像的像素值
            result_pixels[x, y] = (r_result, g_result, b_result)
    
    # 保存结果图像
    result.save(output_image)
def line_deap(top_image, bottom_image, output_image):
    # 打开图像
    top = Image.open(top_image).convert("RGB")
    bottom = Image.open(bottom_image).convert("RGB")
    
    # 确保图像尺寸相同
    if top.size != bottom.size:
        top = top.resize(bottom.size)
    
    # 创建新的图像对象
    result = Image.new('RGB', top.size)
    
    # 获取像素数据
    top_pixels = top.load()
    bottom_pixels = bottom.load()
    result_pixels = result.load()
    
    # 应用划分混合模式
    for x in range(top.width):
        for y in range(top.height):
            # 获取各个通道的颜色值
            r_top, g_top, b_top = top_pixels[x, y]
            r_bottom, g_bottom, b_bottom = bottom_pixels[x, y]
            
            # 应用划分混合模式
            r_result = r_top + r_bottom -255
            g_result = g_top + g_bottom -255
            b_result = b_top + b_bottom -255
            
            # 设置结果图像的像素值
            result_pixels[x, y] = (r_result, g_result, b_result)
    
    # 保存结果图像
    result.save(output_image)
# 示例用法
top_image = "top.png"
bottom_image = "bottom.png"
output_image = "result_image.png"
bottom_image_1 = Image.new('RGB', Image.open(bottom_image).size, (203, 203, 203))
bottom_image_1.save("bottom_1.png")

divide_blend(top_image, bottom_image, output_image)
divide_blend(output_image, "bottom_1.png", "result1.png")

line_deap("result1.png","bottom_1.png","result2.png")

#生成mask图片


def fill_color(img_path):
    img = cv2.imread(img_path)
    mask = cv2.inRange(img, (198, 198, 198), (208, 208, 208))
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 8000:
            mask[labels == i] = 0
    mask = cv2.dilate(mask, None, iterations=2)
    img[mask == 255] = [255, 255, 255]
    img[mask == 0] = [0, 0, 0]
    new_img = Image.fromarray(img)
    new_img.save('mask.png')

if __name__ == '__main__':
    fill_color('result2.png')

#内容识别算法生成最终热力图


def content_aware_fill(image_path, mask_path, result_path):
    # 读取图像和掩码
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 使用Telea方法进行内容识别填充
    result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    # 保存结果
    cv2.imwrite(result_path, result)


content_aware_fill("result2.png", "mask.png", "result3.png")
