

# 1. 文件夹结构介绍
- datasets：训练数据集文件夹
- icon：图标文件
- img：测试数据存放文件夹
- OC_track：目标跟踪
- output：输出文件夹
- weights：权重文件
- yolov5_62：目标检测
- main.py：主程序
- requirements.txt：环境版本
- tools.py：脚本
- UI.py：界面文件
- UI.ui：界面文件


# 2. 环境安装
    见 csdn博客 中的环境安装 https://blog.csdn.net/qq_28949847/article/details/132412611    


# 3. yolov5 目标检测训练
## 训练步骤
    
# 以下都是以 yolov5_62为 root路径

1. 准备数据集（以VOC.yaml数据集为例）

2. 使用datasets文件夹下的voc2v5.py 将xml文件转为txt文件

3. 参照datasets文件夹下创建的示例，创建文件夹结构，并将数据集放入在对应的文件夹下

4. 修改 data文件夹下的VOC.yaml
    # 数据集路径
    path: D:/lg/BaiduSyncdisk/project/person_code/project_self/Yolov5_OCtrack/datasets/airplane
    train: # train数据集
      - train
    val: # val 数据集
      - train
    test: # test 数据集
      - train

    # 修改为自己的类
    names:
      0: airplane


5. 修改 train.py文件夹下的参数
    parser.add_argument('--weights', type=str,
                        default='D:/lg/BaiduSyncdisk/project/person_code/project_self/Yolov5_OCtrack/OCTrack_yolov5/weights/yolov5s.pt',
                        help='initial weights path')
    parser.add_argument('--cfg', type=str, default='./models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/VOC.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')

    ....

6. 点击运行，即可训练

7. 模型保存在 runs 文件夹下
