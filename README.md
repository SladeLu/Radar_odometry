# 基于神经网络的毫米波雷达里程计

## 1. 简介
* 此项目利用毫米波雷达输出的二维平面上的热图,通过相邻帧两两比对,通过神经网络输出之间的二维位姿变换(Δx,Δy,Δw), 并且进行后端优化。

* 项目包括:
    1. 数据集生成器
    2. 数据可视化程序
    3. 模型训练程序
    4. 模型评估程序
    5. 后端优化*

## 2. 运行平台
* 数据采集平台: Turtlebot2 Kobuki 机器人
* 传感器: 
    * 雷达: TI 公司生产的IWR1443/AWR1443
    * IMU: JY901
    * 激光雷达:
* 语言: Python
* 深度学习框架: Keras+Tensorflow
* 依赖库:
        cudatoolkit               10.1.243
        cudnn                     7.6.4 
        h5py                      2.10.0
        keras                     2.3.1
        matplotlib                3.1.1
        numpy                     1.17.4 
        pandas                    0.25.3 
        python                    3.6.9 
        tensorboard               2.0.2
        tensorflow-gpu            2.0.0
        tqdm                      4.41.0


## 3. 文件架构

.

├── evaluation 

├── raw_csv_data

├── hdf5 

├── log_filepath 

├── trained_model 

├── odom_fig

├── modules

│   ├── correlation_layer.py

│   ├── LoadDataset.py

│   ├── loss_fun.py

│   ├── model.py

│   ├── odomtransform.py

│   └── odom_visulizer.py

├── Predict.py

├── Evaluation.py 

├── Dataset_Generater.py

├── Train.py

└── readme.md 

![frame](/md_pic/frame.png)

## 4. 数据集生成器

### 4.1 输入RAW_CSV
输入格式为.csv; 每一行对应一个传感器在某一时刻得到的数据,包括时间戳,数据类型。

|传感器类型|时间(秒)|时间(毫秒)|数据A|数据B|数据C|......|
| ------ | ------ | ------ |------ |------|------ |------ |
imu0  |1575172324|128987072|-0.002130529|-0.004261058|-0.003195793|......|
imu1  |1575172324|222805824|-0.003195793|-0.003195793|-0.006391587|......|
radar  |1575172324|155331028|1955.257784|3342.760835|3501.896772|......|
odom  |1575172324|120308121|0|0|0|......|

当前各类数据格式为：
* IMU
    |imu+序号|时间(秒)|时间(毫秒)|陀螺仪X轴|陀螺仪Y轴|陀螺仪Z轴|加速度计X轴|加速度计Y轴|加速度计Z轴|
    | ------ | ------ | ------ |------ |------|------ |------ |------ |------ |

* 毫米波雷达热图
    |radar|时间(秒)|时间(毫秒)|Data1|Data1|Data2|Data3|......|
    | ------ | ------ | ------ |------ |------|------ |------ |------ |
    雷达行数x列数,项目中为63x128

* 里程计
    |odom|时间(秒)|时间(毫秒)|Data1|Data1|Data2|Data3|......|
    | ------ | ------ | ------ |------ |------|------ |------ |------ |

### 4.2 调用Dataset_Generater.py 生成数据集

1.  主要功能
    * 读取CSV文件并解析传感器原始数据
        `Dataset_generater.csv_release()`
    * 从Ground Truth 的里程计(odom)计算出相邻时间的位姿变换(dof)
        `Dataset_generater.odoms2dof()`
    * 从基于笛卡尔坐标系的位姿变换(Δx,Δy,Δw)计算出对应极坐标系上的位姿变换(Δd,Δρ,Δw)
        `Dataset_generater.__cart2polor()`
    * 对雷达二维热图进行归一化,标准化
        `Mat_Normalize()`
    * 对位姿变换(dof)数据进行归一化,标准化
        `Dataset_generater.__fixdistance()`
        `Dataset_generater.__fixangular()`
    * 划分训练集/测试集, 数据集可以打乱也可以按照原始顺序划分前后两段分别为训练集测试集
        `Dataset_generater.train_test_seperate()`
    * 生成并输出HDF5文件格式的训练集/测试集
        `Dataset_generater.hdf5_generater()`

2. 调用方法
    ```python
    csv_path = '' #csv的路径
    train_rate = 0.8 #训练集占比
    shuffle = True # 是否打乱原有的顺序
    DG_path_tail = '_ensable_shuffle' if DG.shuffle else '_disable_shuffle' #生成的数据集尾缀
    DG = Dataset_generater(csv_path,train_rate, shuffle) # 生成类
    DG.hdf5_generater('./hdf5/Dataset_'+DG_path_tail) # 输出成HDF5文件
    ```

3. 输出HDF5数据集格式

|Keys|格式|大小|内容|
|-------|-------|-------|-------|
|radar_mat|double|(None,2,63,128)|雷达在t与t+1时刻的热图|
|dof_t2t+1|double|(None,3)|雷达在t到t+1时刻之间的位姿变换,坐标格式为笛卡尔坐标系|
|dof_t+12t|double|(None,3)|雷达在t+1到t时刻之间的位姿变换,坐标格式为笛卡尔坐标系|
|dof_polor_t2t+1|double|(None,3)|雷达在t到t+1时刻之间的位姿变换,坐标格式为极坐标系|
|dof_polor_t+12t|double|(None,3)|雷达在t+1到t时刻之间的位姿变换,坐标格式为极坐标系|
|distance|double|(None,1)|雷达在t到t+1时刻之间的位换距离,单位根据标准化缩小倍数而定,/1为m,/10为dm,/100为cm|
|odom_t-1|double|(None,3)|雷达在t-1时刻之间的全局坐标,坐标格式为笛卡尔坐标系|
|odom_t|double|(None,3)|雷达在t时刻之间的全局坐标,坐标格式为笛卡尔坐标系|
|odom_t+1|double|(None,3)|雷达在t+1时刻之间的全局坐标,坐标格式为笛卡尔坐标系|

***

## 5. 构建神经网络模型
1. 模型架构简介
    此模型借鉴与基于深度学习与摄像头的光流神经网络Flownet[^1],将两两帧之间的图像通过CNN网络学习出图片中的光流信息。在[^1]中提出了两种光流神经网络,分别为Flownet-S(Simple)与Flownet-C(Correlation),前者结构简单，所需计算资源小，因其仅使用卷积神经网络；后者加入了作者设计的Correlation_Layer这一层，在后续的论文中，研究者发现拥有复杂结构的后者更容易收敛，因此时间关系，此项目中运用的Flownet均为Flownet-C，如果最后效果不错亦可尝试Flownet-S。
2. Correlation_Layer(modules.correlation_layer.py)
   1. 结构原理      
        代码遵循Keras自定义层级结构格式，在主类中继承`keras.engine.topology.Layer`, 因此只需要重写类方法`__init__()`, `build()`, `call()`, `compute_output_shape()` 即可。 
           
        在本层中，两个特征图f1,f2作为输入, 以每一个位置为中心，上下左右都距离k个单位，构造一个patch，patch的大小为K=2k+1，设来自f1的patch的中心为x1, 来自f2的patch的1中心为x2。计算这两个patch的相关性的办法是：
        $$
        c\left ( x_1,x_2 \right ) = \sum_{o\in \left [ -k,k \right ]\times \left [ -k,k \right ] }\left \langle f_1(x_1+o)),f_2(x_2+o)) \right \rangle
        $$

        计算c(x1,x2)涉及到c* K* K次乘法，比较所有的patch组合涉及到w* w* h* h次计算，所以很难处理前向后向过程。为了计算，引入了最大位移d用于比较，而且在两个特征图中也引入了步长stride。这样通过限制x2的范围，只在D=2d+1 的邻域中计算关联c(x1,x2)。我们用步长s1和s2，来全局量化x1，在以x1为中心的邻域内量化x2。  
        再者,在本层中为两个patch的卷积操作，获得了一个值。因为是数据和数据的卷积，所以这里没有训练参数,所以层级之间基础格式可以使用  `keras.layers.Lambda` 结构直接进行运算。
   2.  初始化参数
        conv_shape 为上一层的输入维度,max_displacement为最大移动距离,stride为步长,输出维度前两维与为输入维度前两维相同,第三维度为(最大移动距离+1)**2,即
    ```python
        Input_dim = (Height,Width,Depth)
        Output_dim = (Height,Width,(Max_displacement+1)**2)
    ```
    3. 调用方法
    ```python
        corr_layer_shape = last_layer.shape[1:] #确认上一层输出的大小
        corr_layer_model = Correlation_Layer(corr_layer_shape,
        max_displacement=4,stride2=2) #生成Correlation_Layer对象
    ```

3. Flownet 
   1. 模型输入  
        输入为雷达二维热图，输入格式为 [(None,128,63,1),(None,128,63,1)],
        位姿真实值即label为(None,3)
   2. 模型结构  
    Flownet沿用Flownet-C的基础上，删去了后半部分上采样的神经网络，只保留前部分的卷积网络与相关层，并且CNN的层数有所减少，以免维度消失。 
    <div align=center><img src="/md_pic/Flownet.png" width="600"/></div> 
    3. 模型输出   
             
    在最后的两层全连接层FC中，之所以分列成两组FC，是因为如果输出的是笛卡尔坐标系，会有两个平移量与一个旋转量(机头偏转值)，即(Δx,Δy,Δw)，位移量归一化效果不好，旋转量能够较好地限定范围，因此两者量纲不完全一致，从 [^2] 中发现如果将旋转值与位移值分开通过不同的全连接层，能够有效提高训练效率与准确性。  

    同理，如果将两个平移量与一个旋转量转换为极坐标下的一个平移量和两个旋转量(Δx,Δy,Δw),则可以更好地约束输出值的范围。
    $$  
    \Delta d= \sqrt{\Delta x^2+\Delta y^2}       \\
    \Delta \rho= \arctan{\frac{\Delta y}{\Delta x}}\\
    \Delta \omega=\Delta \omega  $$
                

    因此输出为输入第一帧与第二帧雷达热力图之间的位姿变换(Δx,Δy,Δw)或(Δd,Δρ,Δw)。

    4. 损失函数(LossFunction)的定义     
        根据文献资料中发现常用的损失函数为MSE(Mean squre error),与MAE(Mean absolute error),两者均可,在训练中可以调整。
        $$
            L_{mse}=(\Delta x_{pred}-\Delta x_{label})^2+(\Delta y_{pred}-\Delta y_{label})^2+(\Delta w_{pred}-\Delta w_{label})^2 \\
            \\
            L_{mae}=|\Delta x_{pred}-\Delta x_{label}|+|\Delta y_{pred}-\Delta y_{label}|+|\Delta w_{pred}-\Delta w_{label}| 
        $$
   

4. Bio-Flownet  
   在文献[^3]中作者提出利用前后帧之间位姿转换数学上的关系可使得模型更加符合问题本质，即由于雷达热图前后帧具有先后顺序，通过神经网络得到的位姿变换T1，与将雷达热图先后顺序颠倒后再通过神经网络得到的位姿变换T2之间存在变换$F`$,使得$T2=F`(T1)$，理论上在一个合理的模型中，T1,T2是一一对应的。
   1. 模型结构  
        本质上与3中的Flownet无区别，相当于是在3外多嵌套了一层中调用了同一个Flownet两次,并且在最后将两者得到的两组维度为3的输出合并。
        ```python
        flownet = Flow_Network()#定义Flownet对象
        forward = flownet([input_layer_pre,input_layer_cur])#将前后两帧正序放入flownet中
        backward = flownet([input_layer_cur,input_layer_pre])#将前后两帧逆序放入flownet中
        output = concatenate([forward,backward])#将两者的输出合并
        ```
   2. 模型输入  
        输入为雷达二维热图，输入格式为 [(None,128,63,1),(None,128,63,1)],
        位姿真实值即label为(None,6)。
        值得注意的是，此处为了避免每一次都要重新计算出新的T2浪费计算时间，因此在生成数据集的时候就已经将T2放入数据集中，之后使用时就可以直接调用。
   3. 模型输出    
        同3.3,输出时通过`concatenate()`将两组(None,3)的Flownet输出合并成为(None,6),分别对应$(Δx,Δy,Δw,Δx',Δy',Δw')$。
   4. 损失函数(LossFunction)的定义
        同3.4
   $$
        L_{mse}=(\Delta x_{pred}-\Delta x_{label})^2+(\Delta y_{pred}-\Delta y_{label})^2+\\
        (\Delta w_{pred}-\Delta w_{label})^2+(\Delta x'_{pred}-\Delta x'_{label})^2+\\
        (\Delta y'_{pred}-\Delta y'_{label})^2+(\Delta w'_{pred}-\Delta w'_{label})^2 \\
    $$

    $$
        L_{mae}=|\Delta x_{pred}-\Delta x_{label}|+|\Delta y_{pred}-\Delta y_{label}|+\\
        |\Delta w_{pred}-\Delta w_{label}|+|\Delta x'_{pred}-\Delta x'_{label}|+\\
        |\Delta y'_{pred}-\Delta y'_{label}|+|\Delta w'_{pred}-\Delta w'_{label}|
    $$

5. Flownet_probability*
   1. 模型设想   
        当前模型应用在后端优化的时候存在一定的问题，即输出较为单一，每次输入只有Δx,Δy,Δw三个输出，很难从概率的角度对模型整体进行优化，联系到在视觉做目标检测中一步法的对每个框加上一个概率的属性，因此想尝试在模型中做出同样的优化。
        <div align=center><img src="/md_pic/Flownet_P.png" width="600"/></div> 
    1. 模型输入  
        同3.3
    2. 模型输出     
        在新模型中FC会输出K*(3+1)组数据,3为(Δx,Δy,Δw),1为输出本组坐标的概率P,K为输出的待选数据的组数，而具体输出哪一组取决于每一组最大的P值。
    $$
        Output=(Δx_i,Δy_i,Δw_i)\\
        i = \max{(P_1,P_2...P_K)}
    $$ 
   2. 损失函数(LossFunction)的定义*         
    $$
        L_{mse}=\sum_{i\in[1,K]} c_i*((\Delta x_{i,pred}-\Delta x_{i,label})^2+(\Delta y_{i,pred}-\Delta y_{i,label})^2+(\Delta w_{i,pred}-\Delta w_{i,label})^2)
    $$
    $$
        \\
        \\
        L_{mae}=\sum_{i\in[1,K]} c_i*(|\Delta x_{i,pred}-\Delta x_{i,label}|+|\Delta y_{i,pred}-\Delta y_{i,label}|+|\Delta w_{i,pred}-\Delta w_{i,label}|)
    $$
    $$ 
            c_i \in R \ or \ c_i =F( P_i)
    $$

## 6.模型的训练
1. 开始训练
   1. 硬件设定      
       1. 指定训练的GPU与Keras Backend：
        ```python
        import os
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        os.environ["CUDA_VISIBLE_DEVICES"] = "13" #指定的GPU序号
        ```
    
   2. 导入数据库    
    为了方便与Data_Generater.py搭配，因此构造了LoadDataset这个类作为工具存放于LoadDataset.py中，使得调用数据集更加便利。
        ```python
        #trainpath,testpath分别为训练集，测试集路径，为list类型，可为空
        #continues参数意思是在训练过程中是否打乱Batch之间的顺序
        LD = LoadDataset(trainpath,testpath,continues=False)
        ```
        模型使用了fit_generator()来进行训练，因此需要搭配从数据集读取batch的函数，为了便于修改于是设定了新方法`get_batch(size)`,其中yield后可以搭配各种从LoadDataset返回的数据，参数size为batch的大小。
        ```python
        def get_batch(size):
            while True:
                #有时候训练过程中需要手动重置模型，
                # 可以将下一行注释
                # model.reset_states()
                m0, m1, dt,dtp1, o_t_1, o_t, o_tplus1= 
                next(LD.get_batch(size,axis_type='polor'))           
                #可以决定是极坐标还是笛卡尔坐标系
                yield [m0, m1], dt
                # label = np.append(d, o_tplus1,axis=-1)
                # res = [m0, m1, o_t_1,o_t], label
                # yield [m0, m1, o_t_1,o_t], label
                # yield [m0, m1], np.append(dt,dtp1,axis=-1)
                # yield [m1, m0], dtp1
        ```
   3. 导入模型
        ```python
        K.clear_session()  # Clear previous models from memory.
        model = Flow_Network() #生成模型类
        adam = Adam() #Optimizer
        # loss_fun = loss_function #自定义的lossfunction
        loss_fun = 'mse'
        model.compile(optimizer=adam, loss=loss_fun, metrics=['accuracy','mse','mae'])
        ```
   4. 设定训练参数  
      需要设定的参数一共有7个:   
      1. 模型种类
      2. 模型优化器Optimizer    
        通常用的是SGD或者Adam,项目中个人偏好使用Adam,对结果影响不大(?)。
        只要在模型compile处修改即可。
      3. 模型损失函数   
        可以使用官方的'mse','mae'作为loss_fun,也可以自定义，为了使程序分工明确，自定义lossfunction位于module.loss_fun.py中
      4. 模型学习率     
        学习率通过方法可以实现自动调节
            ```python
            def lr_schedule(epoch):
            if epoch < 50:
                return 0.001
            elif epoch <= 100:
                return 0.0001
            else:
                return 0.00001
            ```
      5. 数据集是否打乱/随机(shuffle)
      6. 起始轮数(initial_epoch)/中止轮数(epochs)
      7. batchsize:每次放入训练模型的大小       
            建议越大越好(在不爆显存的情况下)
      8. 模型保存   
        设定模型保存的条件，一般使用ModelCheckpoint方法来定义，如下：   
        ```python
            model_checkpoint 
            = ModelCheckpoint(
            filepath,#输出文件保存的位置
            monitor='valid_loss', #设定保存参数的指标，loss或者accuracy                   
            verbose=1,#是否在训练的时候在命令行里显示信息
            save_best_only=True, #是否只保留最好的模型
            save_weights_only=False,#是否只保留模型的参数(True)或者完整的模型(False)
            mode='auto',#loss要保存最小，accuracy保存最高，auto自动识别高低
            period=1)
        ```    
   5. 开始训练
        ```python
        model.fit_generator(
            get_batch(batch_size), 
            epochs=final_epoch, 
            steps_per_epoch=LD.get_batch_step(batch_size),
            callbacks=cbks, 
            shuffle=True, 
            validation_data=(mat_test,label_test),
            initial_epoch=initial_epoch,
            )
        ```

2. 继续训练     
    继续训练相比开始训练增加的步骤只有2个：
    1. 导入未完成的模型
        ```python
        model = load_model(
        model_path, #训练不饱和的模型路径
        custom_objects={
            'tf':tf,
            'Correlation_Layer':Correlation_Layer,
            'loss_function':loss_function,
            'loss_function_2': loss_function_2,
        })
        ```
    2. 设定未完成训练的模型最后的轮数与目标最后的轮数
        ```python
        initial_epoch = 100
        final_epoch = 500
        ```
3. 暂停/中止训练    
    Ctrl+C 即可，如果需要继续训练参见6.2
4. 模型输出     
   最终模型会根据在6.4.8中设定的输出路径存放，每个epoch会有一个本轮最优的模型，当然如果设定了`save_best_only=True`那么如果发现新的模型还没有旧的好的话则不会保存而直接舍弃。
   ![saved_model](/md_pic/saved_model.png)

## 7.模型的评估
1. 模型过程评估(Feature map)(Evaluation.py)
   1. 简介  
      因为在keras的官方接口中，对于一个训练好的模型很难分析其中每一层输出的效果（除非在模型设计之初就写好相关的接口输出），因此为了简化流程，方便使用，利用`model.get_layer(layer_name)`方法，将训练好的模型通过生成子模型的方法，子模型会在适当的地方（我们想要看到输出样式的地方）跳出原先完整的模型，将数据输出后通过`matplotlib`库将featuremap的每一层平铺展示。
   2. 生成子模型    
      在`single_layer_output(data, model, layer_name)`中，因为当使用Bio-Flownet的时候会导致`model.get_layer(layer_name)`得到多个相同名字的layer output而报错，为了不修改原先的模型又不使编程复杂于是加入了`try-catch`，当出现异常即使用Bio-Flownet报错时，执行异常处理，否则执行try。
   3. Feature Map可视化     
      可视化使用`matplotlib`的`pyplot.imshow()`的功能画出图像，图像宽高代表feature map的宽高，每个宽高grid中的亮暗值代表强度，一次输出图像的channel会非常多，因此要在`matplot`画布上划分出MxN的区域，在`SHAPE_PLOT_RULES`中储存的是一个映射用的字典，字典key代表输出有MxN个channel，内容则是（M,N）的排布。例如{512:(8,64)}代表输出512个通道中，在画布上的排列是8,64。因为分辨率的关系排布以N尽可能大显得美观。

      `matplot`存图用的是`pyplot.savefig()`，为了后期处理不失真一般以`.svg`格式存储为矢量图。
      ![evalu](/md_pic/evaluation.svg)
      保存后的路径根据每一层的输出命名为一个文件夹，每一个文件夹里记录了每一step(即放入K组数据则K为step数量)生成的特征图，特征图上的序号表明了第几个通道channel。
    ![filelist](/md_pic/filelist.png)

   4. 多进程优化
      `matplotlib`不支持多线程，因此只能用多进程来优化生成图片的速度。单张图片生成时间小于1s钟，但是当处理多张图片的时候单进程卡顿会导致出图特别迟缓，采用多进程能加快90%的处理速度。进程进度可见IO输出，会有实时处理进度条。具体过程见`Evaluation.process_start()`。
      ![mutiple-process](/md_pic/mutipleprocess.png)
   5. 方法调用
      ```python
      # testset为放入模型中跑的数据
      # model为模型
      process_start(testset,model)
      ```
2. 模型结果评估(Predict.py)
   1. 简介
      这个文件功能比较简单，主要是能够将训练好的模型重新导入后，计算出loss以及accuracy即可，另外有时候需要分别计算模型输出的3个(或6个)的输出分别的loss值，并且将每一个值与groundtruth(label)通过`modules.odom_visulizer`类可视化。
   2. 模型预测
      同6.2.1导入模型后，调用Kera官方的接口即可。
      ```python
      predict = model.predict(testset)
      ```
    3. 可视化       
       在训练的过程中，每一轮epoch的观测参数会同时存在模型文档的目录下的'training_log.csv'文件中，文件参数的设定可以在(Train.py)中的定义修改参数。
        ```python
        csv_logger = CSVLogger(filename=fileps+'training_log.csv',
                        separator=',',
                        append=True)
        ```
        观测值的设定在模型compile的时候定义的。
        ```python
        model.compile(optimizer=adam, loss=loss_fun, metrics=['accuracy','mse','mae'])
        ```
        之后因为loss、accuracy的值不需要实时性，因此用Excel等软件打开'training_log.csv'后可以用软件内置画图软件一拉画出观测值的变化曲线，如下：
        <div align=center><img src="/md_pic/acc.png" width="600"/></div> 
        <div align=center><img src="/md_pic/mae.png" width="600"/></div> 
        <div align=center><img src="/md_pic/mse.png" width="600"/></div> 
      
## 8. 可视化
   1. 简介
      可视化的代码比较杂，因为每个阶段对可视化的要求都不太一样，因此每次写的风格都不是很一致，总体来看不包括Evaluation模型的时候对每一层Featuremap进行可视化输出外，一共有三个需求，一个是对数据集中的里程计进行可视化输出，一个是对之后推算到的以及模型输出的位姿变换的可视化，以及对上者推算到的里程计的可视化。可视化继续使用`matplotlib`库进行可视化操作。
   2. 可视化类的初始化(`modules.odom_visulizer.odom_range_visulizer`)
        使用前先需要声明一个对象:
        ```python
        orv = odom_range_visulizer()
        ```
        之后就可以往类里添加里程计数据:
         ```python
        # 此处需要为加的数据设定颜色color ,例如'red','green'   
        orv.addodom(odoom_1,color)
        ```  
        在orv初始化的时候就可以设定第一支odom，能够简化一小步：
        ```python
        orv = odom_range_visulizer(odoom_1,color)
        ```
   3. 针对数据集里程计的可视化()
        在2的基础上，需要为里程计数据绑定上雷达热图后便可以直接输出图片：
        ```python
        orv.graph_display(radar_mat)
        ```
        效果如下：
        ![odom3](/md_pic/odom_vis1.png)
        左上为里程计在全局地图的投影；右上为雷达热图在极坐标上的图像； 
        左下右下分别为在groundtruth、模型预测输出结果的车头朝向。
   4. 针对位姿变换的可视化
        ```python
        # d1，d2为用于比较groundtruth、模型预测输出结果
        orv.graph_dofcp_only(d1,d2)
        ```
        效果如下：
        ![doffig](/md_pic/dof.svg)
   5. 针对位姿变换推算里程计的可视化
        ```python
        # d1，d2为用于比较groundtruth、模型预测输出结果
        orv.graph_save_odom_only()
        ```
        效果如下：
        ![odom](/md_pic/odomsave.png)

## 9. 转换公式
为了理清计算中的数学关系，特意将一些坐标上的转换都集中在(modules.odomtransform.py)中，需要时候调用即可。
 1. 将全局坐标转换为相对坐标`AxisTransFromGlobal(origin, target)`        
 已知:新原点O 与目标点 T 的全局(起始点)坐标与y正半轴交角分别为$(x_1,y_1,w_1),(x_2,y_2,w_2)$。
 计算: T 以 O 为新坐标系的坐标$(x_2',y_2',w_2')$
 $$
    x_2' =( x_2-x_1)\times\cos(\pi/2 - w_1)+(y_2-y_1)\times\sin(\pi/2 - w_1)\\
    y_2' =-( x_2-x_1)\times\sin(\pi/2 - w_1)+y_2-y_1)\times\cos(\pi/2 - w_1) \\
    w_2' = w_2-w_1
 $$
 2. 将相对坐标转换为全局坐标`AxisTransToGlobal(origin, target)`     
已知:新原点O 的全局(起始点)坐标与y正半轴交角为$(x_1,y_1,w_1)$, T 以 O 为新坐标系的坐标$(x_2',y_2',w_2')$
计算:目标点 T 的全局(起始点)坐标与y正半轴交角$(x_2,y_2,w_2)$
 $$
    x_2'=x_2'\times\sin(\pi/2 - w_1)+y_2\times\cos(\pi/2 - w_1)+x_1 \\
    y_2 =-x_2\times\cos(\pi/2 - w_1)+y_2\times\sin(\pi/2 - w_1)+x_1\\
    w_2 = w_2'+w_1
 $$
 ` AxisTransToGlobal_K(origin, target)`与此方法无异，只是为了适应能在Keras中高效运算而重写的方法。
3. 将四元数转换为欧拉角`Q2Euler(q4)`
$$
    x, y, z, w = q0, q1, q2,  q3\\
    roll = \arctan{\frac{2*(w*x+y*z)}{ 1-2*(x*x+y*y)}}\\
    pitch = \sin{2*(w*y-z*x))}\\
    yaw = \arctan{\frac{2*(w*z+x*y)}{ 1-2*(z*z+y*y)}}
$$

## 注意事项
1. 文中部分代码为节选或者经过一定程度的修改方便理解，具体代码请以工程实际为主。
2. *表示正在工作中尚未完成
3. turtlebot2 里程计车头是x轴，左侧是y轴；正常运算坐标时需要转换为车头是y轴，右侧为x轴，即顺时针个旋转90°。
4. 在跑含有`matplotlib`库的程序时，偶尔发现有的VScode调用多进程以及matplotlib库异常的情况，建议单独开一个命令行运行。
## Reference:
[^1]: Dosovitskiy, Alexey, et al. "Flownet: Learning optical flow with convolutional networks." Proceedings of the IEEE international conference on computer vision. 2015.

[^2]:  Li R, Wang S, Long Z, et al. Undeepvo: Monocular visual odometry through unsupervised deep learning[C]//2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018: 7286-7291.

[^3]: Clark R, Wang S, Wen H, et al. Vinet: Visual-inertial odometry as a sequence-to-sequence learning problem[C]//Thirty-First AAAI Conference on Artificial Intelligence. 2017.