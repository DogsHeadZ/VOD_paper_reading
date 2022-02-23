视频目标检测（VOD）个人综述

包含2016年-2020年的VOD论文个人的简要解读，2021年及其以后的论文不多，可能是陷入瓶颈了。

pixel-level frame-level指在整个帧的层面

box-level instance-level指在检测的目标层面

2016年

2016年应该是VOD产生的一年，共出现了3篇论文，都是关于post-processing的。

## 1、SEQ-NMS FOR VIDEO OBJECT DETECTION(2016ICLR workshop)

这篇论文提出了之后很多论文都会用到的后处理方法Seq-NMS。

![image-20210429150017782](/Users/feihu/Library/Application Support/typora-user-images/image-20210429150017782.png)

原理不复杂，将相邻帧IOU大于某个阈值的box链接起来，然后使用simple dynamic programming algorithm选择分数最大的链，然后对分数进行重新分配如取平均值，最后就得到了新的分数。

## 2、Object Detection from Video Tubelets with Convolutional Neural Networks(2016CVPR)

![image-20210429150449294](/Users/feihu/Library/Application Support/typora-user-images/image-20210429150449294.png)

## 3、T-CNN: Tubelets With Convolutional Neural Networks for Object Detection From Videos

ImageNet Large-Scale Visual Recognition Challenge 2015 winner![image-20210429150501432](/Users/feihu/Library/Application Support/typora-user-images/image-20210429150501432.png)

2和3这两篇论文都是同一个作者写的，思想都差不多，运用了目标跟踪算法、光流图以及各种小trick对框和分数值进行修正，也是一种后处理方式，这里不加介绍了，因为确实用的模块太多太杂了。

2017年

2017年重点在于出现了使用光流来聚合特征的方式。这个时期主要针对2016年的文章无法进行end-to-end的训练进行改进，在训练阶段就用到了temporal信息。

## 4、Deep Feature Flow for Video Recognition(2017CVPR)

关键词：optical-flow, speed，local feature（frame level）

![image-20210429153754705](/Users/feihu/Library/Application Support/typora-user-images/image-20210429153754705.png)

作者用这幅图说明通过key frame加上光流图得到的current frame的特征，和直接从current frame得到的特征差不多。但光流图的计算要比卷积的计算更快，于是提出在sparse key frame上计算特征，然后通过光流图进行特征传播。

![image-20210429154007960](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154007960.png)

​        思想其实很直观，对每一帧首先判断是不是key frame，如果是则直接计算特征，如果不是则通过光流图根据上一个key frame进行特征传播作为当前帧的特征。

​        传播方式为，对current frame i的location p，将其映射到key frame k的![image-20210429154340889](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154340889.png)，其中![image-20210429154424684](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154424684.png)，因为![image-20210429154438371](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154438371.png)通常是一个分数，不能直接对应到特征图上某个坐标点，所以需要进行bilinear interpolation：

![image-20210429154525699](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154525699.png)

![image-20210429154542405](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154542405.png)

关于双线性插值这个博客讲的不错https://blog.csdn.net/u013010889/article/details/78803240。

为了更好地估计特征，引入一个”scale field“ S，The scale field is obtained by applying a “scale function” S on the two frames![image-20210429154732245](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154732245.png)。最后即可得到传播的特征：

![image-20210429154838179](/Users/feihu/Library/Application Support/typora-user-images/image-20210429154838179.png)

Training the flow function is not obligate, and the scale function S is set to ones everywhere.其实这个S有没有应该都影响不大。值得注意的是，这个光流网络首先在Flying Chairs数据集上进行预训练，然后在VID数据集上进行了fine-tuning。

这篇文章对于Key Frame Scheduling没有进行研究，仅采用了简单的fixed key frame scheduling即key frame相隔一个固定的间隔。并指出如何挑选key frame有研究价值（后续确实有文章研究如何挑选key frame）。

文章在实验部分研究了在网络的各种变种，如是否在frame pairs上进行训练，是否fine-tune 光流网络，最后得出结论，需要fine-tune光流网络。

![image-20210429155524933](/Users/feihu/Library/Application Support/typora-user-images/image-20210429155524933.png)

![image-20210429155512604](/Users/feihu/Library/Application Support/typora-user-images/image-20210429155512604.png)

## 5、Flow-Guided Feature Aggregation for Video Object Detection(2017ICCV)

这篇文章和4是同一个作者，思想极为相似，不过这篇注重精度，在每帧上都进行特征的聚集。

关键词：optical flow，accuracy，local feature（frame-level）

![image-20210429155745906](/Users/feihu/Library/Application Support/typora-user-images/image-20210429155745906.png)

![image-20210429160242395](/Users/feihu/Library/Application Support/typora-user-images/image-20210429160242395.png)

从算法图就能看清这个算法，首先也是通过双线性插值得到传播的特征，然后根据传播的特征和原特征得到系数w，将相邻的所有帧根据w进行加权即可得到聚集的特征。**注意系数w是针对每个像素位置的**：

![image-20210502131905329](/Users/feihu/Library/Application Support/typora-user-images/image-20210502131905329.png)

![image-20210502131934987](/Users/feihu/Library/Application Support/typora-user-images/image-20210502131934987.png)

作者在训练时只采用了两个相邻帧即K=2，在推断时选择一个大的K。但是在训练时在一个大范围进行随机采样，这个采样范围和推断时相同，作者将称之为temporal dropout。这其中应该是有内存限制的原因，但从后续对比实验来看训练时采用多大的K影响不大。

![image-20210429160818991](/Users/feihu/Library/Application Support/typora-user-images/image-20210429160818991.png)

![image-20210429160756857](/Users/feihu/Library/Application Support/typora-user-images/image-20210429160756857.png)

## 6、Detect to Track and Track to Detect(2017ICCV)

这篇文章也很有名，在一个网络内同时进行检测和追踪，但我没太看懂，感觉帮助不大，先放着。这个不需要光流的计算，更快。

关键词：accuracy同时提出了一个高speed的版本，track，no optical flow， local feature。

![image-20210429160914196](/Users/feihu/Library/Application Support/typora-user-images/image-20210429160914196.png)

![image-20210429160949818](/Users/feihu/Library/Application Support/typora-user-images/image-20210429160949818.png)

![image-20210429161431276](/Users/feihu/Library/Application Support/typora-user-images/image-20210429161431276.png)

## 7、Object Detection in Videos with Tubelet Proposal Networks

这篇论文和2、3两篇论文仍然是同一作者，用的东西同样很杂，感觉各种东西揉在一起，精度还没他2016年提的T-CNN高，也不想看了。

![image-20210429151058642](/Users/feihu/Library/Application Support/typora-user-images/image-20210429151058642.png)

2018年

2018年就开始百花齐放了

## 8、Fully Motion-Aware Network for Video Object Detection(2018ECCV)

关键词：accuracy，optical flow， local feature（pixel-level and instance-level）。

针对之前的不足：

- As post-processing methods, those rules are not jointly optimized.
- Use **flow estimation** to predict per-pixel motion which is hereinafter referred to as pixel-level feature calibration. However, such pixel-level feature calibration（校准，相当于聚合） approach would be inaccurate when appearance of objects dramatically changes, **especially as objects are occluded**.With inaccurate flow estimation, the flow-guided warping may undesirably mislead the feature calibration, failing to produce ideal results.

创新点：

- 提出了instance-level feature calibration by learning instance movements through time。这个特征对于occlusion是更加鲁棒的，比pixel-level更好。
- 同时学习pixel-level和instance-level calibration。

![image-20210430153419444](/Users/feihu/Library/Application Support/typora-user-images/image-20210430153419444.png)

整个过程分为三步，1、在pixel-level上的特征进行聚合，然后提取proposal；2、先提取proposal，然后在instance-level上的特征进行聚合；3、将两者的聚合输出的position-sensitive score map再进行聚合，得到最终的每个proposal的特征。

8.1 Pixel-level Calibration

第一步提取特征，以及通过光流图提取运动特征：

![image-20210430162758469](/Users/feihu/Library/Application Support/typora-user-images/image-20210430162758469.png)

![image-20210430162848731](/Users/feihu/Library/Application Support/typora-user-images/image-20210430162848731.png)

W是双线性插值，这个和之前的做法都是一样的。注意到这里时直接取平均，而不是像flow-guided那篇计算adaptive weight，这篇文章也说明了是因为他们发现采用直接采用平均效果和计算系数差不多，同时可以节省计算开销。

8.2 Instance-level Calibration

在这里这篇文章的做法比较奇怪，明明可采取和8.1类似思路，直接对proposal的特征进行取平均。但这篇文章选择基于当前帧的位置预测前后帧的位置偏移，来获得前后帧相应的proposal的特征。

现在想明白了确实得这样做：因为不知道前后帧的propasal的对应关系，所以只能通过当前帧获得前后帧的位置偏移，而不是直接从前后帧或者proposal的位置。

具体步骤：

![image-20210430170016014](/Users/feihu/Library/Application Support/typora-user-images/image-20210430170016014.png)

即根据proposal的位置获得其对应的光流。

![image-20210430170246620](/Users/feihu/Library/Application Support/typora-user-images/image-20210430170246620.png)

然后用一个回归网络R获得相邻帧的偏移量，R时一个全连接层的结构。

![image-20210430170333618](/Users/feihu/Library/Application Support/typora-user-images/image-20210430170333618.png)

通过偏移量就可得到当前帧这个proposal在相邻帧的坐标。

![image-20210430170408353](/Users/feihu/Library/Application Support/typora-user-images/image-20210430170408353.png)

最后根据特征就可得到相邻帧的对应proposal的特征，然后同样是取平均获得instance-level聚合的特征。

8.3 Motion Patten Reasoning and Overall Learning Objective

在最后的特征聚合，本文主要提出获得对pixel-level和instance-level特征进行累加的系数的方式。the key issue of combination is to measure the non-rigidity of the motion pattern.就是评价相邻proposal的运动模式。定义

![image-20210430171456485](/Users/feihu/Library/Application Support/typora-user-images/image-20210430171456485.png)

同时定义

![image-20210430171514305](/Users/feihu/Library/Application Support/typora-user-images/image-20210430171514305.png)

一个是运动的剧烈程度，一个是推断occlusion的概率，这两者组成聚合的系数：

![image-20210430171611836](/Users/feihu/Library/Application Support/typora-user-images/image-20210430171611836.png)

总体损失函数：

![image-20210430171832697](/Users/feihu/Library/Application Support/typora-user-images/image-20210430171832697.png)

实验结果：

![image-20210430171734614](/Users/feihu/Library/Application Support/typora-user-images/image-20210430171734614.png)

加上seq-nms可达80.3。

## <span id="9">9、</span>Object Detection in Video with Spatiotemporal Sampling Networks(2018ECCV)

关键词：deformable convolution，accuracy，local feature（frame-level）

针对的缺点：

- post-processing steps cannot be trained end-to-end
- 预测motion（flow network）存在一系列的缺点。1、设计一个有效的光流网络is not trivial；2、训练光流网络需要很多光流数据；3、将光流网络和检测网络结合起来很困难。

创新点：

- 提出在时空上使用deformable convolutions代替光流网络来充分利用时序信息。

9.1 Deformable Convolution

标准的2D卷积：

![image-20210502130224275](/Users/feihu/Library/Application Support/typora-user-images/image-20210502130224275.png)

![image-20210502130233929](/Users/feihu/Library/Application Support/typora-user-images/image-20210502130233929.png)

可变形2D卷积：

![image-20210502130304289](/Users/feihu/Library/Application Support/typora-user-images/image-20210502130304289.png)

因为![image-20210502130328453](/Users/feihu/Library/Application Support/typora-user-images/image-20210502130328453.png)偏移量通常是分数，所以需要进行双线性插值（之前已经介绍过多次了）。https://blog.csdn.net/u013010889/article/details/78803240

9.2 Spatiotemporal Sampling Network

![image-20210502130541685](/Users/feihu/Library/Application Support/typora-user-images/image-20210502130541685.png)

关键步骤：

![image-20210502131135315](/Users/feihu/Library/Application Support/typora-user-images/image-20210502131135315.png)

注意到图2得到是对于support frame进行采样的特征，所以最后还要进行特征聚集（本文提到这一步是受到flow-guided那篇文章的启发）：

![image-20210502131425382](/Users/feihu/Library/Application Support/typora-user-images/image-20210502131425382.png)

![image-20210502131522332](/Users/feihu/Library/Application Support/typora-user-images/image-20210502131522332.png)

S是一个三层的子网络。系数w通过一个softmax层来确保每个像素位置p上所有权重加起来为1：

![image-20210502131636827](/Users/feihu/Library/Application Support/typora-user-images/image-20210502131636827.png)

实验：

在实验中前后support frame的数量K选为13，加行后处理的最佳精度为80.4

## 10、Towards High Performance Video Object Detection(2018CVPR)

这篇文章和文献4、5为同一作者。

这篇文章对于帧间的传播（这个可以将key frame->key frame，key frame->non-key frame根据运动剧烈程度Q集成为一起），key frame和non-key frame间的传播（运用mask对不同位置的像素点，根据运动的剧烈程度选择不同的特征），以及key frame的选择都采用了不同的模块。比较集成，不错。但其实精度提高和速度提升都不大，思想比较好。

关键词：optical flow，accuracy and speed，local feature（frame-level），key scheduling

针对的缺点：

- 对于Sparse feature propagation The propagated features, however, are only approximated and error-prone, thus hurting the recognition accuracy.（即文献4）
- 对于Multi-frame dense feature aggregation, it is much slower due to repeated motion estimation, feature propagation and aggregation.（即文献5）

创新点：

这篇文章针对文献4、5的缺点进行改进，提出一个unified approach，faster, more accurate, and

more flexible。共有三个主要模块：

- sparsely recursive feature aggregation，只在key frames上进行特征的聚集。
- spatially-adaptive partial feature updating，在non-key frames上进行特征的更新。
- temporally-adaptive key frame scheduling，用来代替之前的fixed key frame scheduling。根据特征的质量来预测key frame。

![image-20210504103131843](/Users/feihu/Library/Application Support/typora-user-images/image-20210504103131843.png)

文章首先回顾了作者之前的4、5两篇论文即图1中的(a), (b)，这里不加赘述。文章总结之前的相似性：1）都用到了motion estimation module；2）end-to-end learning。基于这两个原则提出了一个common framework。

10.1 Sparsely Recursive Feature Aggregation

由于相邻帧具有外观的相似性，所以不必在所有帧上进行特征提取，所以这一步只在key frame上进行特征的聚集：

![image-20210504103646031](/Users/feihu/Library/Application Support/typora-user-images/image-20210504103646031.png)

公式(4)和文献5中的略有不同，是其recursive version。

10.2 Spatiallyadaptive Partial Feature Updating

虽然Sparse Feature Propagation加速了检测，然而传播的特征![image-20210504104229568](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104229568.png)是error-prone，因为在相邻帧的某些部分会存在外观的改变。所以本文提出![image-20210504104339796](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104339796.png)来评估![image-20210504104229568](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104229568.png)是否能够很好地近似特征![image-20210504104431063](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104431063.png)。具体为在光流网络上增加一个sibling branch来同时预测![image-20210504104339796](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104339796.png)和motion field![image-20210504104539811](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104539811.png)：

![image-20210504104557422](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104557422.png)

如果![image-20210504104642469](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104642469.png)，说明在像素点p上，![image-20210504104705625](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104705625.png)不能很好地近似![image-20210504104723670](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104723670.png)，所以需要用真正的特征![image-20210504104741982](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104741982.png)进行更新。具体为每一层的每一个像素点对应的特征不同，通过mask U来实现。如果变化剧烈就用自己的特征，如果变化不剧烈就用motion插值特征：

![image-20210504104830925](/Users/feihu/Library/Application Support/typora-user-images/image-20210504104830925.png)

U是一个mask用来选择每个像素点使用哪部分的特征，如果运动不剧烈就使用k->i的特征，如果变化剧烈，就使用nonkey自己的特征。U的梯度更新和Q有关系：

![image-20210504105124087](/Users/feihu/Library/Application Support/typora-user-images/image-20210504105124087.png)

即如果U的梯度和Q的梯度相反，正好与公式6对应，即Q越小反应运动越剧烈，U就要越大，这样公式6后一项就会越小。

对于non-key，同样使用特征聚集：

![image-20210504105032228](/Users/feihu/Library/Application Support/typora-user-images/image-20210504105032228.png)

10.3 Temporallyadaptive Key Frame Scheduling

![image-20210504105201032](/Users/feihu/Library/Application Support/typora-user-images/image-20210504105201032.png)

公式9的含义为通过Q来评估帧的运动剧烈程度，如果这一帧的变化很剧烈，则将其选择为key frame。是一种较为简单但有效的scheduling。

10.4 A Unified Viewpoint

上述10.1和10.2都可放到一个统一的框架内：

![image-20210504105548100](/Users/feihu/Library/Application Support/typora-user-images/image-20210504105548100.png)

![image-20210504105607234](/Users/feihu/Library/Application Support/typora-user-images/image-20210504105607234.png)

总的算法流程图：

![image-20210504105642964](/Users/feihu/Library/Application Support/typora-user-images/image-20210504105642964.png)

训练的损失函数：

![image-20210504105715061](/Users/feihu/Library/Application Support/typora-user-images/image-20210504105715061.png)

公式（12）的第一项鼓励U趋近于1（即鼓励用自己的特征而不是用运动信息进行近似），第二项鼓励U趋近于0，这样就强制被重新计算区域的大小。U的梯度更新和Q有关系，所以U是根据运动剧烈程度而更新的，具有实际意义。同时随机将U设置为0和1。

![image-20210504110006348](/Users/feihu/Library/Application Support/typora-user-images/image-20210504110006348.png)

实验：

![image-20210504110626606](/Users/feihu/Library/Application Support/typora-user-images/image-20210504110626606.png)

精度其实不高，是对于速度的一个折中。

## 11、Optimizing Video Object Detection via a Scale-Time Lattice(2018CVPR)

这篇文章不针对特征，而是针对检测结果即box的坐标来进行传播（这个思想2020ECCV一篇文章有用到）。采用Motion History Image (MHI)代替optical flow进行motion表征。

关键词：Motion History Image，speed，no feature（box-level），key scheduling，global

针对的不足：

- 之前的方法不能很好地兼顾速度与精度，即不能很好地将其结合起来。

创新点：

- 提出了一种具有丰富设计空间的Scale-Time Lattice
- 提出的框架能够兼顾精度与速度
- 提出了能够更有效进行时序传播的网络，提出了一种adaptive scheme for keyframe selection

![image-20210504145918367](/Users/feihu/Library/Application Support/typora-user-images/image-20210504145918367.png)

![image-20210504145935096](/Users/feihu/Library/Application Support/typora-user-images/image-20210504145935096.png)

进行目标检测分为三步：

1. 在sparse key frame上产生目标检测结果
2. 规划从输入节点的检测结果到密集的输出节点的路径
3. 从key frame传播检测结果到intermedieate frame上，并且在尺度上进行优化

11.1 Propagation and Refinement Unit (PRU)

![image-20210504150307300](/Users/feihu/Library/Application Support/typora-user-images/image-20210504150307300.png)

PRU接收连续的两个key frame的检测结果，将其传播到intermeidate frame上，然后优化其输出到另一个尺度上。

![image-20210504150518738](/Users/feihu/Library/Application Support/typora-user-images/image-20210504150518738.png)

本文采用Motion History Image (MHI) as the motion representation。损失函数：

![image-20210504151120740](/Users/feihu/Library/Application Support/typora-user-images/image-20210504151120740.png)

其中![image-20210504151152824](/Users/feihu/Library/Application Support/typora-user-images/image-20210504151152824.png)为从一个时序到另一时序第j个box的偏移量（针对当前帧和前面那个帧的偏移量）。![image-20210504151405737](/Users/feihu/Library/Application Support/typora-user-images/image-20210504151405737.png)为空间尺度上box的偏移量。

11.2 Key Frame Selection and Path Planning

基于一个observation：时序聚集结果趋向于变差当物体很小并且移动很快时（这个和文献10针对的点差不多）。提出an adaptive selection scheme，即应该在存在小而快的物体的帧上挑选key frame，具体为：

![image-20210504151754000](/Users/feihu/Library/Application Support/typora-user-images/image-20210504151754000.png)

当![image-20210504151842569](/Users/feihu/Library/Application Support/typora-user-images/image-20210504151842569.png)低于某个阈值时，一个额外的key frame![image-20210504151909393](/Users/feihu/Library/Application Support/typora-user-images/image-20210504151909393.png)被添加，这个过程在实验中仅进行一次。

挑选好key frame后，本文提出一个简单的方案来规划路径。具体为：

![image-20210504152115482](/Users/feihu/Library/Application Support/typora-user-images/image-20210504152115482.png)

就是对于两个节点的box，在其中间时序的那个帧上进行传播，然后在尺度上优化。这个过程重复两次之后，剩下的帧就用一个线性插值获得box。

11.3 Tube Rescoring

本文最后还添加了一个R-CNN like classifier来重新判断![image-20210504152731065](/Users/feihu/Library/Application Support/typora-user-images/image-20210504152731065.png)：

![image-20210504152849541](/Users/feihu/Library/Application Support/typora-user-images/image-20210504152849541.png)

l是原始检测器给予的标签。

实验：

mAP 79.6% at 20fps，79.0 at 62 fps，还是不错的。

![image-20210504153019646](/Users/feihu/Library/Application Support/typora-user-images/image-20210504153019646.png)

![image-20210504152934498](/Users/feihu/Library/Application Support/typora-user-images/image-20210504152934498.png)

## <span id="12">12、</span>Video Object Detection with an Aligned Spatial-Temporal Memory

关键词：accuracy, global feature(frame-level), no optical flow, memory，LSTM

针对的缺点：

- 利用时序信息的后处理方法是sub-optimal因为时序信息在训练中过被忽略了，所以它不能克服静态检测器的缺点当物体被遮挡了一个长时间
- 在训练中使用时序信息的方法采用固定长度的时序窗，所以其不能对长时间的时序信息建模。While the Tubelet Proposal Network does model long-term dependencies, it uses vectors to represent the memory of the recurrent unit, and hence loses spatial information.

创新点（主要就是建模长时间的时序信息，并进行了特征的空间对齐）：

- 提出Spatial-Temporal Memory Network(STMN) which jointly learns to model and align an object’s long-term appearance and motion dynamics in an end-to-end fashion for video object detection.
- Its core is the Spatial-Temporal Memory Module (STMM), which is a convolutional recurrent computation unit that fully integrates **pre-trained weights** learned from static images.
- To achieve accurate pixel-level spatial alignment over time, the STMM uses a novel MatchTrans module to explicitly model the displacement introduced by motion across frames.

![image-20210504184330725](/Users/feihu/Library/Application Support/typora-user-images/image-20210504184330725.png)

12.1 Spatial-temporal memory module

![image-20210504184529125](/Users/feihu/Library/Application Support/typora-user-images/image-20210504184529125.png)

使用一个类似LSTM的方式来生成和更新memory。为了适配类似LSTM中的gates，对标准的BatchNorm做了两点改动，这里就不细说了。同时，to transfer the weights of the **pre-trained RFCN static image detector**

**into our STMN video object detector**, we make two changes to the ConvGRU。这里的改动也不细说了。

12.2 Spatial-temporal memory alignment

![image-20210504185042622](/Users/feihu/Library/Application Support/typora-user-images/image-20210504185042622.png)

由于目标在运动，所以其特征可能不能对齐。所以提出MatchTrans module to align the spatial-temporal memory across frames。

![image-20210504185238879](/Users/feihu/Library/Application Support/typora-user-images/image-20210504185238879.png)

![image-20210504185331821](/Users/feihu/Library/Application Support/typora-user-images/image-20210504185331821.png)

这个M‘就是对齐后的特征。这里的好处是没有使用光流图，计算更快。后续有些文章用到了这个对齐的方式。

实验：精度80.5

![image-20210504185649571](/Users/feihu/Library/Application Support/typora-user-images/image-20210504185649571.png)

## 13、Mobile Video Object Detection with Temporally-Aware Feature Maps(2018CVPR)

这篇文章主要是追求速度，在移动端和嵌入式设备也能实现高速率的VOD。将SSD作为Mobilenet architecture，将其中的conv替换为depthwise separable convolutions，最后将LSTM插入到SSD中用于提取视频帧的空间和时间信息，据文中描述在cpu上面可以有15fps，不过mAP大概40-50的样子。重点在于修改并融合LSTM，暂时不细看。

关键词：speed, global feature(frame-level), no optical flow, memory，LSTM

![image-20210504185941726](/Users/feihu/Library/Application Support/typora-user-images/image-20210504185941726.png)

——————2018年文章小结：2018年很多文章针对于speed的提升，针对的是光流的缺点。

15、16两篇文章都着重提到了Videos as space-time region graphs(2018ECCV)

14 Relation Distillation Networks for Video Object Detection(2019ICCV)

这篇文章（RDN）在2020年提及并比较的次数很多。

关键词：

针对的缺点：

- 之前的方法没有很好地利用帧间的object relation。

创新点：

- 使用support 帧中的proposal来增强当前帧的proposal的特征。
- 将增强过程分为两步，第一步挑选置信度高的K个proposal，在第二步挑选置信度更高的supportive proposal来反过来增强relation（所以叫Relation distillation）。

14.1 Object Relation Module

这篇文章建立在Relation networks for object detection(2018CVPR)的基础上，故首先对其进行介绍：

![image-20210506105600058](/Users/feihu/Library/Application Support/typora-user-images/image-20210506105600058.png)

通过对M个relation feature进行concate即可得到这个proposal $$R_i$$的增强特征：

![image-20210506105736759](/Users/feihu/Library/Application Support/typora-user-images/image-20210506105736759.png)

14.2 Relation Distillation Networks

![image-20210506105157636](/Users/feihu/Library/Application Support/typora-user-images/image-20210506105157636.png)

图中refenrence(r)即当前帧，support(s)即相邻帧。

本文提到，自然的做法就是计算reference frame和supportive frame中所有物体间的关系，然而这样做计算量很大，而且由于supportive proposal数量的增加或导致relation learning的不稳定，所以将整体分为两步（图例其实很好地说明了算法，这里就简单引用原文）：

**Basic stage：**![image-20210506110126733](/Users/feihu/Library/Application Support/typora-user-images/image-20210506110126733.png)

如图所示得到$$R^{r1}$$，其中![image-20210506110302868](/Users/feihu/Library/Application Support/typora-user-images/image-20210506110302868.png)堆叠了$$N_b$$个relation module (iterate the relation reasoning in a stacked manner equipped with $$N_b$$ object relation modules to better characterize the relations across all the supportive proposals with regard to reference proposals)。具体为对于第k个relation module和第i个reference proposal：

![image-20210506110553209](/Users/feihu/Library/Application Support/typora-user-images/image-20210506110553209.png)

公式4最后一次迭代的输出即为basic stage的输出![image-20210506110751336](/Users/feihu/Library/Application Support/typora-user-images/image-20210506110751336.png)

**Advanced stage：**basic stage探索了reference 和 supportive proposal的关系，然而没有探索supportive proposal和supportive proposal间的关系。

![image-20210506111029500](/Users/feihu/Library/Application Support/typora-user-images/image-20210506111029500.png)

![image-20210506111044699](/Users/feihu/Library/Application Support/typora-user-images/image-20210506111044699.png)

整体算法流程图：

![image-20210506111136415](/Users/feihu/Library/Application Support/typora-user-images/image-20210506111136415.png)

14.3 Box Linking with Relations

这里提出了一种后处理方法，首先指出the object relations between detection boxes are not fully studied for box linking。然后提出formulate the post-processing of box linking as an optimal path finding problem。由于box linking独立地被应用于每个类别，所以在这里省去类别的标签。

![image-20210506111537408](/Users/feihu/Library/Application Support/typora-user-images/image-20210506111537408.png)

![image-20210506111713853](/Users/feihu/Library/Application Support/typora-user-images/image-20210506111713853.png)

核心为寻找一条最优链，而其中与之前方法如seq-nms不同之处为利用了proposal间的关系系数w，然后对判别框进行重打分。

实验：

![image-20210506112138109](/Users/feihu/Library/Application Support/typora-user-images/image-20210506112138109.png)

15、Leveraging Long-Range Temporal Relationships Between Proposals for Video Object Detection(2019ICCV)

和[文献12](#12)的目的类似，想要建立long-term relation，但是本文用单独一段提到文献12是在local neighborbhood上进行传播，而本文的intsance-level的方法不需要locality，同时文献12对于快速移动的物体检测有困难因为local feature propagation。2018年很多在key frame上进行传播的不纳入此类，因为他们是偏向于speed的方法。

本文意思是通过增加间隔s来捕捉long-term信息（因为本文的方法没有空间的约束，那其实也可以通过打乱顺序捕捉long-term信息），并且不需要很多的support frame，因为instance已经很多了？

![image-20210508110418561](/Users/feihu/Library/Application Support/typora-user-images/image-20210508110418561.png)

关键词：accuracy and speed，global feature(instance-level)，no optical flow，nonlocal，gragh

针对的缺点：

- 后处理的方法和基于光流的方法显著降低了速度
- 基于key frame的方法精度较低

创新点：

- 提出proposal-level temporal relation block学习外观相似性并使用support frame特征增强目标特征
- a method of applying the relation block to incorporate **long-term** dependencies from multiple support frames in a video
- 建立关系gragh，并添加有监督的图训练

方法：

![image-20210508103753004](/Users/feihu/Library/Application Support/typora-user-images/image-20210508103753004.png)

定义：

![image-20210508104111322](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104111322.png)

![image-20210508104154857](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104154857.png)

分为三步：

![image-20210508104255403](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104255403.png)

通过添加Graph loss来约束图矩阵使得相同类别的实例在图矩阵中对应更高的值：

![image-20210508104411859](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104411859.png)

最后对图矩阵进行normalize，得到![image-20210508104655877](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104655877.png)，这样做可以这样进行分析：

![image-20210508104746093](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104746093.png)

本文还补充介绍了一些小trick：

Causal and Symmetric modes：

![image-20210508104846516](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104846516.png)

![image-20210508104852815](/Users/feihu/Library/Application Support/typora-user-images/image-20210508104852815.png)

Single Model and Frozen Support Model settings：

在single model setting中，both support frame和target frame都送入网络中进行end-to-end训练，然而这里会存在两个问题：

![image-20210508105039628](/Users/feihu/Library/Application Support/typora-user-images/image-20210508105039628.png)

所以提出Frozen Support Model setting：

![image-20210508105233262](/Users/feihu/Library/Application Support/typora-user-images/image-20210508105233262.png)

但实验显示在一个模型上学习特征事实上更好。

实验：本文做了大量的实验，感兴趣可以看看。In Table 4 we report 80.6 mAP, and the speed of 10 FPS on Titan X Pascal GPU for the causal model with only one relation block at fc7 layer.

![image-20210508105640547](/Users/feihu/Library/Application Support/typora-user-images/image-20210508105640547.png)



16、Sequence Level Semantics Aggregation for Video Object Detection(2019ICCV)

也是想建立global long-term的信息。对于[文献12](#12)，本文提到文献12通过memory模块进行信息传递，本文则不需要，而且本文是在instance-level上进行特征聚集。

这篇文章聚集不同frame的proposal的特征，用的就是简单的proposal的特征相似性。优点在于将其作为一个聚类问题考虑，不需要考虑时序，不需要光流，能用到全局信息（就是对frame进行随机采样）。思想比较简单，不过这篇文章出的比较早，2020年很多都是这样做的。

关键词：accuracy，global feature(instance-level)，no optical flow

针对的缺点：

- 后处理方法不能end-to-end训练
- end-to-end的特征聚集方法需要使用光流图或者实例追踪等，这些方法依赖于精确的运动评估，是矛盾的。同时这些方法大多仅聚集nearby frames。

创新点：

- 将视频目标检测看作是一个sequence level multishot detection problem，并且提出对于VID任务的全局聚类的视角。
- introduce a simple but effective Sequence Level Semantics Aggregation (SELSA) module to fully utilize video information。

方法较简单，只有两个方程

通过余弦相似性来衡量两个proposal间的语义相似性：

![image-20210506101803263](/Users/feihu/Library/Application Support/typora-user-images/image-20210506101803263.png)

![image-20210506101923649](/Users/feihu/Library/Application Support/typora-user-images/image-20210506101923649.png)

得到语义相似性之后，就可根据其进行多个proposal的特征聚集：

![image-20210506102259362](/Users/feihu/Library/Application Support/typora-user-images/image-20210506102259362.png)

聚集的特征就输入后续的检测网络进行classfication and bounding box regression。

文章还用了一节A Spectral Clustering Viewpoint分析其与classic spectral clustering算法的关系，从降低类内方差的视角来看待本文提出的SELSA。这里不加分析。

实验：

在训练时，![image-20210506102637923](/Users/feihu/Library/Application Support/typora-user-images/image-20210506102637923.png)

![image-20210506102658307](/Users/feihu/Library/Application Support/typora-user-images/image-20210506102658307.png)

同时本文分析不同的采用策略并**采用了打乱顺序的采样的方式**。并进行实验发现加上后处理反而会降低精度，说明本文已经captured the full sequence level information，不需要额外的后处理进行增强了。

同时采用了数据增强：

![image-20210506102833997](/Users/feihu/Library/Application Support/typora-user-images/image-20210506102833997.png)

最终实验精度还是很不错的：

![image-20210506102923973](/Users/feihu/Library/Application Support/typora-user-images/image-20210506102923973.png)

17、Object Guided External Memory Network for Video Object Detection(2019ICCV)

关键词：accuracy and speed，global feature(pixel-level and instance-level)，no optical flow，memory

![image-20210506143904750](/Users/feihu/Library/Application Support/typora-user-images/image-20210506143904750.png)

针对的缺点(分为dense methods和recurrent methods)：对于[文献12](#12) STMN，提到其传播的是整个特征图。

- dense methods需要储存邻近的超过20帧的**完整**的特征图，导致low storage- efficiency。
- recurrent methods将过去的信息都压缩为一个特征图，这样当当前的帧退化和剧烈变化时会导致interrupted。

创新点：

- 通过object guided **hard** attention(hard attention is non-differentiable and is trained with reinforcement learning or predefined based on human knowledge)来改进storage-efficiency，并且通过external memory传播时序信息来解决long-term dependency。
- 设计object guided external memory network，和novel write and read操作来储存和精确地传播multi-level特征。Soft and hard attentions are respectively used in our read and write operations。

方法：在pix-level和instance-level两个层面改善特征

![image-20210506151311009](/Users/feihu/Library/Application Support/typora-user-images/image-20210506151311009.png)

![image-20210506151448849](/Users/feihu/Library/Application Support/typora-user-images/image-20210506151448849.png)

17.1 External memory matrices

![image-20210506151705986](/Users/feihu/Library/Application Support/typora-user-images/image-20210506151705986.png)

17.2 Read operation

![image-20210506152102387](/Users/feihu/Library/Application Support/typora-user-images/image-20210506152102387.png)

这个特征应该是在把长宽维度展开为一维，所以变成了n x c，n=h x w

和一般的attention机制类似，分为三步(非常典型)：

![image-20210506152148135](/Users/feihu/Library/Application Support/typora-user-images/image-20210506152148135.png)

The challenge is that the external memory size varies, leading to big difference of signal strength in the aligned memory. 为此推出scale parameters![image-20210506152313560](/Users/feihu/Library/Application Support/typora-user-images/image-20210506152313560.png)to ensure that the features in aligned memory are in proper magnitude and signal strength of present/memory features in the output aggregated feature maps is balanced.

![image-20210506152414850](/Users/feihu/Library/Application Support/typora-user-images/image-20210506152414850.png)

![image-20210506152500273](/Users/feihu/Library/Application Support/typora-user-images/image-20210506152500273.png)

对j进行累加才是1，估计是经过实验发现需要对这些系数加个限制效果才好，然后这样进行解释。

在实践中使用了多头机制：

![image-20210506152538840](/Users/feihu/Library/Application Support/typora-user-images/image-20210506152538840.png)

![image-20210506152557184](/Users/feihu/Library/Application Support/typora-user-images/image-20210506152557184.png)

17.3 Write operation

**For pixel level features** write is **guided by detected objects**. 对于obejct b特征：

![image-20210506154007730](/Users/feihu/Library/Application Support/typora-user-images/image-20210506154007730.png)

就是坐标位于物体b的bounding box内的特征（维度为c*1）被认为是物体b的特征，其想要被挑选入external memory中需要满足两个条件：

![image-20210506154359139](/Users/feihu/Library/Application Support/typora-user-images/image-20210506154359139.png)

![image-20210506154532468](/Users/feihu/Library/Application Support/typora-user-images/image-20210506154532468.png)

在实践中，当没有物体被检测出来时，we also include object-irrelevant features into the external memory.

![image-20210506154638879](/Users/feihu/Library/Application Support/typora-user-images/image-20210506154638879.png)

![image-20210506154735272](/Users/feihu/Library/Application Support/typora-user-images/image-20210506154735272.png)

For **instance level features**, write operate用当前帧的proposal 特征代替所有之前的intance memory：

![image-20210506154917675](/Users/feihu/Library/Application Support/typora-user-images/image-20210506154917675.png)

17.4 Efficient Detection with External Memory

为了提高检测效率，本文同样挑选key frame和non key frame（就是固定间隔采样），对non-key frame进行下采样。因为下采样后只是特征图的大小变了，但通道数没变，所以本文的momory机制仍然能够对齐使用：

![image-20210506155121067](/Users/feihu/Library/Application Support/typora-user-images/image-20210506155121067.png)

实验（没有进行后处理）：

![image-20210506155236374](/Users/feihu/Library/Application Support/typora-user-images/image-20210506155236374.png)



18、Fast Object Detection in Compressed Video(2019ICCV)

目前已有不少在LSTM的结构上进行改进来进行视频目标检测的了，**在这上面是否还能做下去？**

关键词：speed，local feature(pixel-level)，compressed video，FFmpeg，H.264，Pyramidal Feature，memory，LSTM

针对的缺点:

- 用CNNs处理密集的帧很耗费时间。
- 对于FlowNet需要额外的计算时间，并且忽略了a video is generally stored and transmitted in a compressed data format。The codecs split a video into I-frames (intra-coded frames) and P/B-frames (predictive frames). An I-frame is a complete image while a P/B-frame **only holds the changes** compared to the reference frame.

创新点：

- 加快提取**压缩视频**的特征，只需要对reference frame进行完整的识别，对于predictive frame只需使用轻量的网络。
- 不需要额外的网络来建模帧的motion信息，而是直接使用视频中已有的motion vectors and residual errors。
- Propose a pyramidal feature attention that enables the memory network to propagate features from multiple scales. It helps to detect objects across different scales.

方法：

![image-20210506172702838](/Users/feihu/Library/Application Support/typora-user-images/image-20210506172702838.png)

使用H.264进行视频的压缩，压缩后的视频包含两种类型的帧：

![image-20210506172936075](/Users/feihu/Library/Application Support/typora-user-images/image-20210506172936075.png)

这个memory网络包含两个主要模块：pyramidal feature attention ![image-20210506184156603](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184156603.png)和motion-aided LSTM ![image-20210506184211553](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184211553.png)。总体的流程：

![image-20210506184231361](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184231361.png)

18.1 Pyramidal Feature Attention

![image-20210506184402110](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184402110.png)

首先将各层的特征转换为相同的维度：

![image-20210506184457716](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184457716.png)

然后在通道维度压缩特征：

![image-20210506184534794](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184534794.png)

最后用生成的scale descriptors来生成注意力权重从而结合不同尺度的特征：

![image-20210506184650489](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184650489.png)

![image-20210506184701931](/Users/feihu/Library/Application Support/typora-user-images/image-20210506184701931.png)

16.2 Motion-aided Memory Network

对于压缩图像介绍这里没太看懂，就不加以介绍了。本文采用FFmpeg来提取每个P-frame的motion vectors and residual errors，并将其reszie为特征图的大小。

本文使用LSTM来传递特征，并对传统的LSTM做了两点修改：One is the **motion vector aided feature warping** and another is **residual error based new input**.

**motion vector aided feature warping**：

![image-20210506185112910](/Users/feihu/Library/Application Support/typora-user-images/image-20210506185112910.png)

warping operation就是普遍的双线性插值：

![image-20210506185217798](/Users/feihu/Library/Application Support/typora-user-images/image-20210506185217798.png)

**residual error based new input**：

本文使用residual errors作为新的输入，在获得warped特征和新的输入之后，memory机制如下：

![image-20210506185359452](/Users/feihu/Library/Application Support/typora-user-images/image-20210506185359452.png)

实验：

![image-20210506185627216](/Users/feihu/Library/Application Support/typora-user-images/image-20210506185627216.png)

17、Progressive Sparse Local Attention for Video Object Detection

[文献12](#12) 先提出类似的思想，本文提到与文献12的不同：Different from [37], our approach focuses on a sparse neighborhood and softmax normalization is utilized to better establish spatial correspondence. We improves both speed and accuracy while STMN improves accuracy at the expense of runtime.

进行特征传播的方式除了光流图、文献12的MatchTrans，还有Nonlocal。好像本文是nonlocal的一种变体。Basically, Match-Trans computes the propagation weights by accumulating all similarity scores in the local region while Nonlocal considers all positions. By contrast, PSLA uses a progressively sparse local region.本文在实验部分充分比较了三者的实验结果来说明本文所提方法的优势。

关键词：speed，global feature(pixel-level)，no optical flow，attention，nonlocal

针对的缺点：

- 光流网络显著地增加了模型的大小
- 光流网络仅仅建立了两幅图像local pixel的联系。Directly transferring the flow field to high-level features may introduce artifacts because it ignores the transformation that happens from layer to layer in the network。
- 高维度特征中的一个像素点对应图像中的很多像素，光流网络不能捕获这样一个大的位移

创新点：

- 提出Progressive Sparse Local Attention (PSLA)来建立特征图间的对应关系，不需要额外的光流图，所以减少了模型的参数并且提高了精度
- 基于PSLA，提出了两个模块Recursive Feature Updating (RFU) and Dense Feature Transforming (DenseFT)来分别model temporal appearance and enhance the feature representation of non-key frames

方法：

![image-20210508094956137](/Users/feihu/Library/Application Support/typora-user-images/image-20210508094956137.png)

从图2可以看出，本文对于key frame和non-key frame采用不同的特征提取网络，采用的也是在key frame上传递并更新特征，然后再传递到non-key frame上，这个框架很普遍了。

17.1 Progressive Sparse Local Attention

![image-20210508095401581](/Users/feihu/Library/Application Support/typora-user-images/image-20210508095401581.png)

设计PSLA的motivation是光流图的边缘分布集中于0附近，这说明用于计算对应关系权重的feature cell能被限制在a neighborhood with progressively sparser strides：

![image-20210508095520634](/Users/feihu/Library/Application Support/typora-user-images/image-20210508095520634.png)

具体的公式部分看原文就很清晰了：

![image-20210508100059691](/Users/feihu/Library/Application Support/typora-user-images/image-20210508100059691.png)

17.2 Recursive Feature Updating（在key frame和key frame间传递特征）

![image-20210508100527594](/Users/feihu/Library/Application Support/typora-user-images/image-20210508100527594.png)

![image-20210508100711266](/Users/feihu/Library/Application Support/typora-user-images/image-20210508100711266.png)

17.3 Dense Feature Transforming（在key frame和non-key frame间传递特征）

![image-20210508100929981](/Users/feihu/Library/Application Support/typora-user-images/image-20210508100929981.png)

对于为什么要加上一个Transform Net本文有解释：

![image-20210508101037338](/Users/feihu/Library/Application Support/typora-user-images/image-20210508101037338.png)

![image-20210508101050476](/Users/feihu/Library/Application Support/typora-user-images/image-20210508101050476.png)

实验（对于文献12和non-local做了大量的对比实验说明本文的优势）：

![image-20210508101158952](/Users/feihu/Library/Application Support/typora-user-images/image-20210508101158952.png)

“+”号代表增加了后处理Seq-NMS。



18、Video Object Detection with Locally-Weighted Deformable Neighbors(2019AAAI)

关键词：accuracy and speed，local feature(pixel-level)，no optical flow，memory，deformable

其实[文献9](#9)也用到了可形变卷积，但本文并没有提到文献9。

针对的缺点：

- 光流网络比较耗时
- 训练光流模型需要大量的数据

创新点：

- 用可形变卷积的方式，通过预测weight和offset来传递特征
- 通过memory机制在帧间传递特征，并在key frame 和 key frame间更新momory

方法：

18.1 Locally-Weighted Deformable Neighbors

![image-20210507130245352](/Users/feihu/Library/Application Support/typora-user-images/image-20210507130245352.png)

介绍从key frame 到 non-keyframe的方式：

![image-20210507130310571](/Users/feihu/Library/Application Support/typora-user-images/image-20210507130310571.png)

这个可形变卷积前面已经讲过多次了。简而言之，要学习kernel weight W和偏移量kernel offset，这样就可以根据公式4将特征进行传播。这个本质上也和文献12、17差不多，融合了不同位置的特征（这个文章写到是受到Non-Local Operator的启发，其实文献12的思想也差不多）。

18.2 Memory-Guided Propagation Networks Inference

![image-20210507130754777](/Users/feihu/Library/Application Support/typora-user-images/image-20210507130754777.png)

注意这节讲的是inference而不是训练。整个inference过程图例讲的很清晰了，对于key frame同时提取低维和高维特征，对于non-key frame仅提出低维特征。需要注意的是，对于key frame to key frame，$$W_0$$同时生成weights和offests。对于key frame to non-key frame，$$W_1$$只生成weights。

18.3 Memory-Guided Propagation Networks Training

![image-20210507131617963](/Users/feihu/Library/Application Support/typora-user-images/image-20210507131617963.png)

这一节讲训练过程。训练时每个batch取三帧：

![image-20210507131754709](/Users/feihu/Library/Application Support/typora-user-images/image-20210507131754709.png)

![image-20210507131806503](/Users/feihu/Library/Application Support/typora-user-images/image-20210507131806503.png)

图例也讲的很明白了，不过有一个问题是这样的三张图像key frame和non-key frame的顺序不会像inference那样，non-key frame严格在key frame之后（因为non-key frame只考虑其之前的key frame）。

18.3 Feature Correlation

We use **correlation** which performs multiplicative path comparisons between two neighbor low-level feature maps **as the input of Weight Predictor Network**.算是小trick

![image-20210507132627156](/Users/feihu/Library/Application Support/typora-user-images/image-20210507132627156.png)

对于特征聚集啥的也用了一些小trick，这里不加详述，总的算法流程：

![image-20210507133303716](/Users/feihu/Library/Application Support/typora-user-images/image-20210507133303716.png)

实验：

![image-20210507133748106](/Users/feihu/Library/Application Support/typora-user-images/image-20210507133748106.png)

19、 Detect or Track: Towards Cost-Effective Video Object Detection/Tracking(2019AAAI)

这篇文章用一个孪生网络来判断是否仅需要对前面那个key frame进行追踪（追踪速度更快），还是要重新进行检测（检测速度较慢）。==这个孪生网络的思想或许可以用到。==

关键词：

针对的缺点：

- 基于目标追踪的VOD算法依赖于traker的质量，会陷入局部最优。

创新点：

- 提出scheduler network来选择进行追踪还是进行检测。并将其和强化学习联系起来，scheduler network is shown to be a generalization of Siamese trackers and a special case of RL
- 通过追踪的质量来决定当前帧是否应该为key frame

方法：

![image-20210507170232608](/Users/feihu/Library/Application Support/typora-user-images/image-20210507170232608.png)

19.1 The Scheduler Network in DorT

![image-20210507170750181](/Users/feihu/Library/Application Support/typora-user-images/image-20210507170750181.png)

The scheduler network in DorT aims to determine to detect or track given a new frame by estimating the quality of the tracked boxes.因为这篇文章追求速度，所以这个网络要是effcient。对于特征，不重头训练一个网络，而是重新利用部分tracking网络的特征：

![image-20210507170909655](/Users/feihu/Library/Application Support/typora-user-images/image-20210507170909655.png)

下标l说明是第l层特征。其实就是一个判别网络，决定是进行追踪还是进行检测。

19.2 Training Data Preparation

因为数据集中没有对追踪质量的评估，所以本文simulate the tracking process between two sampled frames and label it as detect (0) or track (1) in a strict protocol.

![image-20210507171912063](/Users/feihu/Library/Application Support/typora-user-images/image-20210507171912063.png)

这篇文章之后还分析了其与SiamFC Tracker、强化学习的关系，需要的话可以看论文。

实验（这篇文章的实验有点没太看懂，主要是对scheduler网络的实验，后续可以再看看）：

![image-20210507172248337](/Users/feihu/Library/Application Support/typora-user-images/image-20210507172248337.png)

2019年利用了long-term的信息，即global feature，光流图基本没人用了，取而代之的是其他的特征融合方式。

2020年

## 20、Memory Enhanced Global-Local Aggregation for Video Object Detection(2020CVPR)

这篇文章mega通过[Relation Networks for Object Detection 2018CVPR]的relation module充分进行了global和local层面的聚集，并引入了memory机制，使得涉及的帧更多。是一篇很综合很工程的文章。

关键词：accuracy，global feature and local feature(instance level)，memory

针对的缺点：

- 之前的方法没有**同时考虑local和global的信息**
- 之前的“global”仅仅是一个一个更大的local range，而本文的local是指整个video

创新点：

- 采用一个multi-stage的结构，结合了global semantic information（与当前帧相隔很远的帧的信息，这篇文章通过打乱视频帧的顺序随机得到相隔较远帧的信息）和local localization information（与当前帧相邻帧的信息）。但是这些信息依然是limited。
- 提出Long Range Memory模块使得当前帧能够得到更多之前帧的信息

![image-20210407152847150](/Users/feihu/Library/Application Support/typora-user-images/image-20210407152847150.png)

20.1 Relation Module

对于每个box $b_i$，通过relation module来增强其semantic特征表达[Relation Networks for Object Detection 2018CVPR]：

![image-20210407153149860](/Users/feihu/Library/Application Support/typora-user-images/image-20210407153149860.png)

$w_{ij}^m$为$b_i$和$b_j$的semantic特征相似性，是一个multi-head attention思想。（1）的意思是，通过B中的所有box来加强$b_i$的semantic特征表达。具体为计算当前proposal和其它prolposal间的关系w，然后和其它proposal的特征进行加权。

![image-20210407153249529](/Users/feihu/Library/Application Support/typora-user-images/image-20210407153249529.png)

​		M代表Head的个数，即算了M次（如果不用multi-head的思想只需计算一次即可），将这M次的特征与原始特征进行concat，就构成了$b_i$的augmented feature。

20.2 Global-Local Aggregation for ineffective problem

Globa aggregation将global feature G聚集到local feature L：

![image-20210407153759117](/Users/feihu/Library/Application Support/typora-user-images/image-20210407153759117.png)

​	  $N_g$是一个栈结构：

![image-20210407153851690](/Users/feihu/Library/Application Support/typora-user-images/image-20210407153851690.png)

​	 即不断地用全局特征G增强local feature $L$ ，最后的输出即为$L^g$。

​	然后进行local aggregation，将上一步的输出作为这步的输入：

![image-20210407154203627](/Users/feihu/Library/Application Support/typora-user-images/image-20210407154203627.png)

​	同样是一个栈结构：

![image-20210407154239127](/Users/feihu/Library/Application Support/typora-user-images/image-20210407154239127.png)

​	最后的输出视作C，被用作RCNN后面的分类和框回归

20.3 Long Range Memory for insufficient problem

Global-Local Aggregation仍然不能聚集足够多数量的帧的信息，于是引入LRM模块来聚集更多帧的信息。具体为每当$I_{k-1}$帧被检测完成后，会将其在local aggregation阶段local feature中最开始帧的特征送入LRM中（每一层栈的特征都会送进去），就类似于一个滑窗的形式，每检测完一帧就滑一帧。这样local aggregation变为：

![image-20210407154747690](/Users/feihu/Library/Application Support/typora-user-images/image-20210407154747690.png)

![image-20210407154759963](/Users/feihu/Library/Application Support/typora-user-images/image-20210407154759963.png)

​		相比于原来的local aggregation，这个引入了M信息，使得能够记录更多帧的信息。

![image-20210407154855178](/Users/feihu/Library/Application Support/typora-user-images/image-20210407154855178.png)

实验：

![image-20210508182111675](/Users/feihu/Library/Application Support/typora-user-images/image-20210508182111675.png)

21、CenterNet Heatmap Propagation for Real-time Video Object Detection(2020ECCV)

关键词：speed，local feature(pixel level)，centernet

针对的缺点：

- 两阶段的目标检测算法较慢
- adapting existing temporal information merging methods to one-stage detectors is challenging or even infeasible.（因为这些方法多用到了一阶段目标检测算法没有的RoIs）

创新点：

- 提出了一种基于一阶段检测算法**CenterNet的heatmap传播方法**

这篇文章目的是提高检测速率，实现实时检测。两阶段目标检测算法很慢，普通的一阶段目标检测算法在VOD上很多时候是infeasible。所以提出采用anchor-free的一阶段目标检测算法CenterNet来进行检测。

![image-20210407203517098](/Users/feihu/Library/Application Support/typora-user-images/image-20210407203517098.png)

CenterNet输出一个进行了下采样的热力图$\hat{Y}\in[0,1]^{\frac wR\times\frac hR\times c}$,c为类别数。热力图代表了目标中心的类别，同时输出O（修正位置）和S（object size）：

![image-20210407204549743](/Users/feihu/Library/Application Support/typora-user-images/image-20210407204549743.png)

获得了热力图，需要进行heatmap propagation：

![image-20210407204840373](/Users/feihu/Library/Application Support/typora-user-images/image-20210407204840373.png)

这幅图说的挺清楚了，简而言之就是预测t+1帧时，将t的各个object的热力图进行扩充（b），然后进行overlap（c），最后将其叠加到t+1帧的热力图上，即可进行预测。不过其具体步骤涉及的东西比较多。主要在于叠加时系数的计算，这里不加详述了，直接看论文。

这篇文章主要就是将CenterNet用到了VOD上，并设计了一个讲道理的传播方法，实现了实时的VOD。

实验：

![image-20210508183502583](/Users/feihu/Library/Application Support/typora-user-images/image-20210508183502583.png)

效果没the Scale-Time Lattice framework好，但是说把他们仍然采用faster-rcnn，自己的方法能够结合上去。

22、Mining Inter-Video Proposal Relations for Video Object Detection(2020ECCV)

这篇文章认为之前的方法只考虑了视频内部（intra-video）的关系，而没有考虑各个不同视频间（inter-video）的关系，导致其不能很好区分相似的类别（如花纹相似的猫和狗）。于是这篇文章提出了分别在video-level和proposal-level（即框的level）进行Triplet Selection（即对于一个target，构建同一个种类的正样本，和不同种类的负样本，一起构成triplet）。

关键词：accuracy，global feature(instance level)

针对的缺点：

- 之前的方法只考虑了视频内部（intra-video）的关系，而没有考虑各个不同视频间（inter-video）的关系，导致其不能很好区分相似的类别（如花纹相似的猫和狗）

创新点：

- 提出Inter-Video Proposal Relation method, which can effectively leverage inter-video proposal relation to learn discriminative representations for video object detection
- Integrate intra-video and inter-video proposal relation modules in a unified framework

方法：

![image-20210407205535034](/Users/feihu/Library/Application Support/typora-user-images/image-20210407205535034.png)

22.1 Video-Level Triplet Selection

这一步选择和target video即当前video的support video。选择的原则是正样本要选择the most dissimilar support video in the class，负样本要选择is the most similar support video in the other classes，这样就构成了hard training（这样训练更有效果）。For each video, we randomly sample one frame as target frame t, and sample other T − 1 frames as support frames, e.g., frame t − s and frame t + e in Fig. 2.具体做法就是将各个video随机选择的T帧进行特征提取，然后执行global average pooling along spatial and temporal dimensions，这样就得到了一个C维的video representation，根据特征的cosine similarity即可计算不同video的相似性。这一步就可到video triplet：

![image-20210407210734032](/Users/feihu/Library/Application Support/typora-user-images/image-20210407210734032.png)

22.2 Intra-Video Proposal Relation

这一步就是通在各个video内部进行通常的spatio-temporal proposal aggregation，即将每个video挑选的T帧的各个box进行聚集。

![image-20210407210412021](/Users/feihu/Library/Application Support/typora-user-images/image-20210407210412021.png)

22.3 Proposal-Level Triplet Selection

这一步是选择合适的候选框的triplet，和第一步基本一样，也是根据特征的余弦相似性来的，得到proposal triplet：

![image-20210407210721811](/Users/feihu/Library/Application Support/typora-user-images/image-20210407210721811.png)

22.4 Inter-Video Proposal Relation

这一步聚集这个proposal triplet的信息：

![image-20210407210841710](/Users/feihu/Library/Application Support/typora-user-images/image-20210407210841710.png)

同时引入两个损失函数：

![image-20210407210913493](/Users/feihu/Library/Application Support/typora-user-images/image-20210407210913493.png)

其中$L_{detection}$是传统的检测损失（比如包含bbox regression和object classification）。$L_{relation}$ 是一个简洁的 triplet-style的损失函数：

![image-20210407211019156](/Users/feihu/Library/Application Support/typora-user-images/image-20210407211019156.png)

即使得正负样本更具有区分度，从而使得target样本和他们也更有区分度。

这篇论文聚集了不同video的信息，核心在于triplet的思想。

实验：

![image-20210508184041955](/Users/feihu/Library/Application Support/typora-user-images/image-20210508184041955.png)

23、Video Object Detection via Object-levelTemporal Aggregation(2020ECCV)

这篇文章认为在更高的level上（如object-level即bounding-box level）上做聚集相比于在feature level上是更有效的。这样做的好处是，1、box的维度相比于特征的维度降低了；2、不再需要重新训练特征提取器。

之前有篇文章也不需要用到特征，也是在object level上进行聚集。以目标检测器和追踪器的结果组成的四维向量![image-20210520160327407](/Users/feihu/Library/Application Support/typora-user-images/image-20210520160327407.png)作为object level的特征。

本文的相关工作部分写的很详细，分了很多个类别。

关键词：speed，no feature(object level)，reinforcement learning，keyframe scheduler，tracker，LSTM

针对的缺点：

- 之前的方法在feature level上做聚集，存在维度高，需要重新训练等缺点

创新点：

- 结合目标检测算法和目标追踪算法在object/bounding-box level上进行时序聚集
- present **adaptive keyframe schedulers** using simple heuristics and RL training with diverse video sequences

方法：

![image-20210509101026106](/Users/feihu/Library/Application Support/typora-user-images/image-20210509101026106.png)

![image-20210509101348892](/Users/feihu/Library/Application Support/typora-user-images/image-20210509101348892.png)

思想较为清晰，在key frame上进行目标检测，然后通过tracker传递到non-key frame上。目标检测和目标追踪符号定义：

![image-20210509101316313](/Users/feihu/Library/Application Support/typora-user-images/image-20210509101316313.png)

然后介绍基于LSTM的特征聚集方式（在key frame上进行特征聚集，在non-key frame上就直接进行tracking）：

![image-20210509101456553](/Users/feihu/Library/Application Support/typora-user-images/image-20210509101456553.png)

除了基于LSTM的聚集方式，本文同时还介绍了另一种heuristic module，相较于LSTM速度更快但精度降低：

![image-20210509101905640](/Users/feihu/Library/Application Support/typora-user-images/image-20210509101905640.png)

本文重点在于接下来的基于强化学习的key frame scheduling，目前不是很感兴趣，先空着：

![image-20210509101940839](/Users/feihu/Library/Application Support/typora-user-images/image-20210509101940839.png)

实验：

采用两种目标追踪器KCF和SiamFC，对于KCF不需要预训练；对于SiamFC，在GOT-10K数据集上进行训练，然后直接用。

![image-20210509102156862](/Users/feihu/Library/Application Support/typora-user-images/image-20210509102156862.png)

精度：

![image-20210509102243742](/Users/feihu/Library/Application Support/typora-user-images/image-20210509102243742.png)

虽然精度不高，但在CPU上能取得极高的速度。

24、Learning Where to Focus for Ecient Video Object Detection

和Progressive Sparse Local Attention for Video Object Detection（2019ICCV）思想基本一样，写作风格、introduction甚至实验都差不多，都是一个实验室的，只不过一作不一样但通讯作者是同一个人。对key和non key frame采用不同的骨干网络，达到速度和精度都不错的效果。

关键词：speed，global feature(pixel-level)，attention，nonlocal

针对的缺点：

- 光流网络显著地增加了模型的大小
- Optical flow cannot represent the correspondences among high-level features accurately
- 光流图的计算很耗时

创新点：

- LSTS module is proposed to propagate the high-level feature across frames, which could calculate

  the spatial-temporal correspondences accurately.Dierent from previous approaches, LSTS treat the offsets of sampling locations as parameters and the optimal offsets would be learned through back-propagation guided by bounding box and category supervision

- SRFU and DFA module are proposed to model temporal relation and enhance feature representation, respectively

方法：和之前那篇实在是很像，最大的不同是本文对于每个像素点关联的location**是可以学习的**（又用到了双线性插值，因为其可能是分数），而之前那篇就采取的就是周围递进的location。本文location的更新也是梯度下降：

![image-20210509152646258](/Users/feihu/Library/Application Support/typora-user-images/image-20210509152646258.png)

其余部分基本一致，这里就不加赘述了。

![image-20210509152159880](/Users/feihu/Library/Application Support/typora-user-images/image-20210509152159880.png)

![image-20210509152227102](/Users/feihu/Library/Application Support/typora-user-images/image-20210509152227102.png)

![image-20210509152239122](/Users/feihu/Library/Application Support/typora-user-images/image-20210509152239122.png)

实验：性能提升也不大

![image-20210509152750642](/Users/feihu/Library/Application Support/typora-user-images/image-20210509152750642.png)



25、Dual Semantic Fusion Network for Video Object Detection(2020MM)

关键词：accuracy，global feature(pixel level and instance level)

针对的缺点：

- 之前的方法在frame level或instance level上做聚集，而没有同时考虑两者

创新点：

- 提出dual semantic fusion network，同时在frame level和instance level上进行multi-granularity semantic fusion（所以是dual）
- introduce **a geometric similarity measure** into the proposed dual semantic fusion network along with the widely used **appearance similarity** measure to alleviate the information distortion caused by noise during the fusion process
- 从information theory的角度来解释VOD过程，然后give a detailed analysis to show the effectiveness of the proposed dual semantic fusion network

方法：

![image-20210509104702807](/Users/feihu/Library/Application Support/typora-user-images/image-20210509104702807.png)

从图中可以看出，主要包括两个模块，分别在frame-level和instance-level上进行fusion。

25.1 Frame-level Semantic Fusion

其实还是基于non-local的老一套，对于frame-level，在通道维度上进行相似性的计算：

![image-20210509105306308](/Users/feihu/Library/Application Support/typora-user-images/image-20210509105306308.png)

25.2 Instance-level Semantic Fusion

对于instance-level，每个instance的特征已经是一维的了，所以直接在instance上的特征上进行相似性的计算：

![image-20210509110259996](/Users/feihu/Library/Application Support/typora-user-images/image-20210509110259996.png)

![image-20210509110330097](/Users/feihu/Library/Application Support/typora-user-images/image-20210509110330097.png)

这个embedding function将低维的位置相似性信息转换为高维信息。借鉴了Relation networks for object detection。

25.3 An Information Theory Viewpoint

这节从信息论的角度进行了分析，这里暂不关心。

实验（打乱顺序效果最好，这其实也是用了全局信息）：

![image-20210509110940522](/Users/feihu/Library/Application Support/typora-user-images/image-20210509110940522.png)

26、Exploiting Better Feature Aggregation for Video Object Detection(2020MM)

关键词：accuracy，global feature(instance level，也是通过打乱顺序实现global)，attention，spatial

针对的缺点：

- 之前的方法在只考虑了物体的时序依赖性，而没有考虑空间的联系
- 之前的方法在时序域上对所有support proposals进行特征聚集，而没有考虑其是否属于同一类（这有点和gragh的思想类似）
- 没有对特征进行对齐

创新点：

- 同时考虑时序和空间的特征聚集，即spatial-temporal feature aggregation network
- 提出class homogeneity限制，只使用相同类别的support proposal来增强target proposal
- 提出feature alignment module来对齐特征

方法：

![image-20210509143433971](/Users/feihu/Library/Application Support/typora-user-images/image-20210509143433971.png)

相比于一般的时序上的特征聚集，本文首先采用了一个简单的对于proposal的判别器，来聚集相同类别的proposal。进行时序聚集之后，还增加了一个空间聚集（仅在target frame上进行空间聚集）。在实现特征聚集时加入了特征对齐模块。

26.1 Temporal aggregation module

![image-20210509143723278](/Users/feihu/Library/Application Support/typora-user-images/image-20210509143723278.png)

通过两个全连接层的网络分别处理support获得key，处理target获得query：![image-20210509144209075](/Users/feihu/Library/Application Support/typora-user-images/image-20210509144209075.png)。通过query和key的相似性获得系数：

![image-20210509144251392](/Users/feihu/Library/Application Support/typora-user-images/image-20210509144251392.png)

对于value，本文不增加一个映射层而是直接采用原始的值作为value。同时特征对齐在support的特征进行，后续会讲这个模块。

![image-20210509144439553](/Users/feihu/Library/Application Support/typora-user-images/image-20210509144439553.png)

此处借用了残差机制，为了使得拼接的support特征和target特征维度一致，在每层最开始时利用1*1的卷积来降低输入特征的维度。

26.2 Spatial relation module

这个模块聚集target frame上不同物体的特征。The Spatial Relation Unit (SR Unit) in this module is an extension of the TA Unit。其不同之处在于系数的计算不一样，SRU还用到了box的全局坐标信息：

![image-20210509144946380](/Users/feihu/Library/Application Support/typora-user-images/image-20210509144946380.png)

也是先将低维坐标信息映射到高维，然后计算关系。最后系数的计算就被增强了：

![image-20210509145046042](/Users/feihu/Library/Application Support/typora-user-images/image-20210509145046042.png)

26.3 Feature alignment module

![image-20210509145108964](/Users/feihu/Library/Application Support/typora-user-images/image-20210509145108964.png)

对于target frame的坐标(m,n)将其复制拓展为原特征的大小，然后support feature和target feature都经过一个1*1的卷积获得![image-20210509145552115](/Users/feihu/Library/Application Support/typora-user-images/image-20210509145552115.png)和![image-20210509145600024](/Users/feihu/Library/Application Support/typora-user-images/image-20210509145600024.png)然后计算correlation map：

![image-20210509145349833](/Users/feihu/Library/Application Support/typora-user-images/image-20210509145349833.png)

26.4 Class constraint

这个是为了只聚集相同类别的proposal的特征，其实就是增加了一个损失函数：

![image-20210509150101771](/Users/feihu/Library/Application Support/typora-user-images/image-20210509150101771.png)

26.5 Video level aggregation

这一节讲的是因为这个模型不需要考虑顺序，所以采用随机取样，效果更好。即通过打乱序列的顺序获得全局信息。

实验：

![image-20210509150201769](/Users/feihu/Library/Application Support/typora-user-images/image-20210509150201769.png)



27、Temporal Context Enhanced Feature Aggregation for Video Object Detection

关键词：accuracy，local feature(pixel-level)，spatial，deformable convolution

这篇文章在Temporal Stride Predictor部分这个和我将物体分为fast，medium，slow三种模态分别进行分析有点类似。不过这里是根据**模态选择不同的stride**，而我是想建立不同的模型。而且这里是建立的网络是输入图像的差异，直接输出分数，标签是真实的IoU。我想的是输入两张图像，建立孪生网络来输出分数。不过这个思想是类似的。这篇文章有个观点：

![image-20210509151232285](/Users/feihu/Library/Application Support/typora-user-images/image-20210509151232285.png)

他认为有些方法对于序列的顺序没有要求表明其没有受利于时序信息，感觉很扯，人家这明明用到的是全局信息。

不过这篇文章感觉还是有点水，特别是介绍部分根本没介绍清楚，后面写的也不咋滴，精度也不高，AAAI的文章不好看。主要就是一个Temporal Stride Predictor来预测相邻图像的stride，然后一个时空网络，而且他这个空间网络也很勉强（感觉就是一个特征维度上的注意力机制），不如Exploiting Better Feature Aggregation for Video Object Detection中的空间模块讲道理，不过也可以解释为对于空间的两个不同的方面。感觉时空网络都要被用烂了。感觉就是将各个模块拼接起来，而且各个模块都没有什么新意。

![image-20210509150449385](/Users/feihu/Library/Application Support/typora-user-images/image-20210509150449385.png)

![image-20210509150454349](/Users/feihu/Library/Application Support/typora-user-images/image-20210509150454349.png)

![image-20210509150954229](/Users/feihu/Library/Application Support/typora-user-images/image-20210509150954229.png)

![image-20210509151404347](/Users/feihu/Library/Application Support/typora-user-images/image-20210509151404347.png)

实验很糟糕，要精度没精度，要速度没速度，唯一的优势好像是聚集的帧数更少，但这也不能显著降低速度啊...

![image-20210509151539042](/Users/feihu/Library/Application Support/typora-user-images/image-20210509151539042.png)

