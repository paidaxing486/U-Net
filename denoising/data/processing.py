import os
import cv2

def image_cut_save(img, left, upper, right, lower, save_path):
    """
        所截区域图片保存
    :param path: 图片路径
    :param left: 区块左上角位置的像素点离图片左边界的距离
    :param upper：区块左上角位置的像素点离图片上边界的距离
    :param right：区块右下角位置的像素点离图片左边界的距离
    :param lower：区块右下角位置的像素点离图片上边界的距离
     故需满足：lower > upper、right > left
    :param save_path: 所截图片保存位置
    """
    cropped = img[upper:lower, left:right]

    # 保存截取的图片
    cv2.imwrite(save_path, cropped)


#获得需要处理的图片
root_path = 'C:/Users/sch12/Desktop/denoising/data/example/'
save_path = 'C:/Users/sch12/Desktop/denoising/data/test_example/'
for filename in os.listdir(root_path):
    print(filename)
    filenames = os.path.join(root_path + filename)
    fname = str(filename.strip()[:-4])
    img = cv2.imread(filenames)
    img2 = cv2.resize(img,(2048,2048))
    save_filenames = os.path.join(save_path + fname + '.jpg')
    left, upper, right, lower = 512, 768, 1024, 1280
    image_cut_save(img2, left, upper, right, lower, save_filenames)
