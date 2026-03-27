#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main() {
    // ====================== 任务1：读取图片 ======================
    string img_path = "/home/lzy/cv-course/build/test.jpg";

    // 检查文件是否存在
    FILE* f = fopen(img_path.c_str(), "r");
    if (f == NULL) {
        cout << "错误：文件不存在！路径: " << img_path << endl;
        return -1;
    }
    fclose(f);

    Mat img = imread(img_path, IMREAD_COLOR);
    if (img.empty()) {
        cout << "图片读取失败！可能是文件格式不支持" << endl;
        return -1;
    }
    cout << "图片读取成功！" << endl;

    // ====================== 任务2：输出图像信息 ======================
    int height = img.rows;
    int width = img.cols;
    int channels = img.channels();
    string dtype = "8U";  

    cout << "\n图像基本信息" << endl;
    cout << "图像尺寸（宽×高）: " << width << " × " << height << endl;
    cout << "图像通道数: " << channels << endl;
    cout << "像素数据类型: " << dtype << endl;

    // ====================== 任务3：显示原图 ======================
    imshow("Original Image", img);
    waitKey(0);
    destroyAllWindows();

    // ====================== 任务4：转灰度图并显示 ======================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);

    imshow("Gray Image", gray_img);
    waitKey(0);
    destroyAllWindows();

    // ====================== 任务5：保存灰度图 ======================
    string save_path = "/home/lzy/cv-course/build/gray.png";
    imwrite(save_path, gray_img);
    cout << "\n灰度图已保存至: " << save_path << endl;

    // ====================== 任务6：像素访问 + 裁剪 ======================
    // 输出像素值
    if (channels == 3) {
        Vec3b pixel = img.at<Vec3b>(502, 205);
        cout << "\n坐标 (502,205) 的像素值: "
             << "B=" << (int)pixel[0] << " "
             << "G=" << (int)pixel[1] << " "
             << "R=" << (int)pixel[2] << endl;
    } else {
        uchar pixel = gray_img.at<uchar>(502, 205);
        cout << "\n坐标 (502,205) 的像素值: " << (int)pixel << endl;
    }

    // 裁剪
    const int CUT_WIDTH = 400;
    const int CUT_HEIGHT = 400;

    int cut_w = min(CUT_WIDTH, width);
    int cut_h = min(CUT_HEIGHT, height);

    Mat cut_img = img(Rect(0, 0, cut_w, cut_h));

    cout << "\n裁剪区域：左上角(" << cut_w << ", " << cut_h << ")" << endl;
    cout << "裁剪后尺寸：宽" << cut_img.cols << " × 高" << cut_img.rows << endl;

    imshow("Cut Image", cut_img);
    waitKey(0);
    destroyAllWindows();

    // 保存裁剪图
    string cut_save_path = "/home/lzy/cv-course/build/cut.png";
    imwrite(cut_save_path, cut_img);
    cout << "裁剪区域已保存至: " << cut_save_path << endl;

    return 0;
}