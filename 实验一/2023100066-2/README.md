
# OpenCV 基础图像处理与单步调试实验实验

## 1. 项目介绍
本项目使用Vscode + OpenCV4 实现基础图像处理功能，包括图片读取、信息查看、灰度转换、图像裁剪、显示与保存等操作,将代码利用c++进行改写，并对程序进行单步调试。

## 2. 运行环境
- Vscode/c++
- Linux/wsl

## 3. 主要功能

1. 读取本地图片并判断文件是否存在
2. 输出图像尺寸、通道数、数据类型
3. 显示原始图像
4. 将彩色图像转为灰度图并显示
5. 保存灰度图像
6. 获取指定坐标像素值
7. 裁剪图像左上角区域并保存

## 4. 核心代码与说明

### 4.1 编译环境配置
```c++
//在vscode工作区键入以下指令,打开编译的配置文件
.vscode/tasks.json
//完成配置
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "build main",
            "type": "shell",
            "command": "g++",
            "args": [
                "-g",
                "build/main.cpp",
                "-o",
                "build/main",
                "`pkg-config",
                "--cflags",
                "--libs",
                "opencv4`"
            ],
            "group": {
                "kind": "build",
                "isDefault": false
            },
            "problemMatcher": [
                "$gcc"
            ]
        }
    ]
}
```
### 4.2 调试环境配置
```c++
//在vscode工作区键入以下指令,打开调试的配置文件
.vscode/launch.json
//完成配置
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug main",
            "type": "cppdbg",
            "request": "launch",
            "program": "${workspaceFolder}/build/main",
            "args": ["input.txt", "123"],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "gdb",
            "miDebuggerPath": "/usr/bin/gdb",
            "preLaunchTask": "build main"
        }
    ]
}
```
### 4.3 图像读取
```c++
Mat img = imread(img_path, IMREAD_COLOR);
// 判断图片是否加载成功
if (img.empty()) {
    cout << "图片读取失败！可能是文件格式不支持" << endl;
    return -1;
}
```

### 4.4 获取图像信息
```c++
int height = img.rows;       // 图像高度
int width = img.cols;        // 图像宽度
int channels = img.channels();// 通道数
string dtype = "8U";         // 像素数据类型
```
### 4.5 转为灰度图
```c++
Mat gray_img;
cvtColor(img, gray_img, COLOR_BGR2GRAY);
```
### 4.6 保存图像
```c++
imwrite(save_path, gray_img);    // 保存灰度图
imwrite(cut_save_path, cut_img); // 保存裁剪图
```
### 4.7 图像裁剪
```c++
// 裁剪 400x400，自动防止越界
int cut_w = min(400, width);
int cut_h = min(400, height);
Mat cut_img = img(Rect(0, 0, cut_w, cut_h));
```
## 5. 使用说明
1. 将测试图片命名为 test.jpg 放入 build 目录
2. 编译命令：Ctrl+Shift+B（VS Code）
3. 运行 / 调试：F5
4. 图片窗口弹出后，按任意键关闭当前窗口并继续执行
5. 运行前要创建相应编译与调试的配置
   
## 6. 作者信息
1. 作者：李智阳
2. 日期：2026年3月27日
