import os
import random
import json

def read_split_data(root: str,test_rate: float = 0.2):
    """
    divide the dataset
    :param root: the path to the dataset(train_dataset and test_dataset)
    :param test_rate: the proportion of the test_dataset
    :return: the image path and label corresponding to the training set and test set
    """
    random.seed(42)  

    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # traverse the folders, one folder corresponds to one kind
    texture_class = [cla for cla in os.listdir(root)
                    if os.path.isdir(os.path.join(root, cla))]  
    # sort it
    texture_class.sort()
    
    # generate category names and their corresponding indexes
    class_indices = dict((k, v) for v, k in enumerate(texture_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(root+'/class_indices.json', 'w') as json_file:
        json_file.write(json_str)
        
    # stores the image path of the dataset along with the label index
    train_images_path = []      
    train_images_label = []     

    test_images_path = []       
    test_images_label = [] 
    
    # stores the total number of samples for each category
    every_class_num = []
    
    # supports file suffix names
    supported = [".jpg", ".JPG", ".png", ".PNG"]  

    # start!!!
    for cla in texture_class:
        cla_path = os.path.join(root, cla)
        
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        
        # gets the index corresponding to the category
        image_class = class_indices[cla]
        
        # record the number of each category
        every_class_num.append(len(images))
        
        random.shuffle(images)
        
        test_path = images[: int(len(images) * test_rate)]
        # test_path = images[int(len(images) * val_rate):int(len(images) * (test_rate+val_rate))]

        for img_path in images:
            if img_path in test_path: 
                test_images_path.append(img_path)
                test_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for testing.".format(len(test_images_path)))
    
    '''
    # 绘制不同类别图片在 训练集 、验证集 、 测试集中的数量
    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()
    '''

    return train_images_path, train_images_label, test_images_path, test_images_label

'''
def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = r"D:\00\00_papers\1_Magid_Texture-Based_Error_Analysis_for_Image_Super-Resolution_CVPR_2022_paper(1)\30_related data set and others\datasets\dtd\images\class_indices.json"
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.savefig('./data.jpg')
            plt.imshow(img.astype('uint8'))
        plt.show()
        '''
