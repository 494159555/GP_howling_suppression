# 第1章 绪论

## 1.1 研究背景与意义

声音是人类信息传递的最自然媒介之一，随着社会信息化程度的不断加深，扩声系统已成为确保语音信息在各类场景下高效、清晰传递的重要基础设施，广泛应用于教室、会议室、体育馆、演出场馆等公共场所[1][2]。典型的扩声系统由麦克风、功率放大器和扬声器三大部分组成，其基本工作流程为：麦克风拾取声源信号，经功率放大器将电信号放大后，由扬声器将电信号还原为声波向听众辐射。然而，当扬声器和麦克风处于同一声场环境中时，扬声器辐射的声波会经过空间的多次反射与传播后再次被麦克风拾取，由此形成了一个从麦克风经放大器、扬声器到空间声场再回到麦克风的闭合声学回路。在该闭环系统中，若某一频率的信号满足环路增益大于等于1且相位满足正反馈条件（即奈奎斯特稳定准则），则该频率的信号就会在回路中不断循环放大，产生自激振荡，最终形成刺耳的高频正弦音——即声反馈啸叫，简称啸叫（Howling）[3]。

啸叫现象的存在对扩声系统造成了多方面的严重危害。首先，从听觉体验来看，啸叫产生的尖锐音会严重掩盖正常语音信号，导致语音可懂度急剧下降，影响信息传递的效率和准确性，尤其在教学、会议等对语音清晰度有严格要求的场景中，啸叫问题尤为突出。其次，从设备安全角度来看，啸叫形成的高强度信号会驱动扬声器长时间工作在高功率状态下，可能导致扬声器振膜过载甚至烧毁功放器件，造成不可逆的硬件损坏[4]。此外，啸叫现象也限制了扩声系统的最大可用增益（MAG），即系统在不产生啸叫的前提下所能提供的最大声学放大倍数，这进一步制约了扩声系统在大空间、远距离场景下的应用效果。

鉴于上述问题，研究有效的声反馈啸叫抑制（Acoustic Howling Suppression, AHS）技术具有极其重要的现实意义和应用价值。从技术发展的历史脉络来看，啸叫抑制技术经历了从人工干预到模拟电路控制，再到数字信号处理（DSP）自动控制的发展历程。目前主流的软件啸叫抑制方法主要包括以下几类：

第一类是移频法（Frequency Shifting, FS）与移相法（Phase Modulation, PM）[5]。该方法通过在信号传输链路中对信号施加微小的频率偏移或相位调制，破坏啸叫产生的正反馈相位条件，从而抑制啸叫。移频法实现简单、计算量小，是较早被商用的啸叫抑制方案之一。然而，频率偏移操作本身会对原始信号引入可感知的失真，尤其在音乐信号处理中，移频会导致音调的微小偏移和和声关系的破坏，严重影响音质。因此，移频法主要适用于对音质要求不高的语音扩声场景。

第二类是陷波抑制法（Notch-based Howling Suppression, NHS）[6][7]。该方法的核心思路是通过实时检测信号中出现的啸叫频率点，然后在对应频率处设置窄带陷波滤波器进行定点抑制。陷波抑制法的有效性高度依赖于前端啸叫检测算法的准确度和响应速度。在实际应用中，啸叫频率点的误检和漏检问题时有发生：误检会导致正常信号被错误衰减，造成音质损失；而漏检则会导致啸叫无法被及时抑制。此外，陷波器的插入会在一定程度上扭曲目标信号的频谱包络，甚至可能引入新的啸叫频率点[3]。当多个啸叫点同时出现时，大量陷波器的叠加使用会进一步加剧信号失真。

第三类是自适应反馈消除法（Adaptive Feedback Cancellation, AFC）[8][9]。这是目前理论上最为完善的啸叫抑制方案。该方法通过自适应滤波器对扬声器到麦克风的声反馈路径进行在线估计，并从麦克风信号中减去估计的反馈分量，从而在信号层面消除声反馈。理论上，若反馈路径能被完美估计，则声反馈信号可被完全消除。然而，在实际情况中，目标信号与反馈信号来源于同一声源，两者之间存在高度相关性，这种强相关性会导致自适应滤波器的估计偏差。为了解决这一问题，研究人员引入了多种去相关技术，如注入探测噪声、非线性处理等，但这些去相关手段本身又不可避免地降低了语音质量[8]。此外，自适应算法（如LMS、NLMS）的收敛速度与稳态误差之间存在固有的矛盾，难以同时满足快速跟踪反馈路径变化和低失调量的需求。

综上所述，传统啸叫抑制方法虽然在工程实践中得到了广泛部署，但均存在不同层面的局限性：移频法以牺牲音质为代价换取稳定性，陷波法依赖于不够鲁棒的啸叫检测且会引入频谱失真，自适应反馈消除法则受限于信号相关性和去相关处理带来的音质损失。这些共性不足促使研究者探索新的技术路径来突破传统方法的瓶颈。

近年来，深度学习技术在计算机视觉、自然语言处理和语音信号处理等领域取得了突破性进展，为音频处理领域带来了新的范式[10]。特别是在语音增强（Speech Enhancement）、语音分离（Speech Separation）和声学回声消除（Acoustic Echo Cancellation, AEC）等任务中，基于深度神经网络的方法已经展现出超越传统信号处理方法的显著优势[11][12]。深度学习强大的非线性建模能力使其能够从大量数据中自动学习复杂的信号映射关系，无需显式地对物理过程进行数学建模。这一特性对于啸叫抑制问题具有重要意义——啸叫信号的产生涉及复杂的房间声学路径、多次反馈叠加和非线性失真等过程，这些因素使得传统方法难以进行精确的数学建模，而深度学习可以通过数据驱动的方式隐式地学习这些复杂关系。

在众多深度学习架构中，U-Net卷积神经网络因其在图像分割领域的成功而备受关注[13]。U-Net采用编码器-解码器结构，通过编码器逐步提取多尺度的高层语义特征，再通过解码器逐步恢复空间分辨率，同时利用跳跃连接将编码器中的低层细节特征传递给解码器，从而在保持精确定位能力的同时捕获全局上下文信息。这一架构特性使其非常适合处理频谱级别的信号估计任务。事实上，U-Net及其变体已被成功应用于音频源分离、语音增强和音乐信号处理等任务中[14][15]，证明了其在频谱掩膜估计方面的有效性。

将U-Net架构应用于啸叫抑制问题的核心思路是：将啸叫抑制任务抽象为一个频域掩膜估计问题。具体而言，对带啸叫的音频信号进行短时傅里叶变换（STFT）获得其时频表示，将归一化的对数幅度谱作为U-Net的输入，网络通过学习带啸叫频谱与干净频谱之间的映射关系，输出一个时频掩膜（Mask），将该掩膜与原始带啸叫信号的频谱相乘即可得到增强后的频谱，再通过逆STFT重构为时域信号。这种端到端的学习范式无需显式地进行啸叫频率点检测或声反馈路径估计，从根本上避免了传统方法所面临的诸多困难。

基于上述分析，本研究具有重要的理论意义和实践价值。在理论层面，本研究将探索深度学习技术应用于声反馈啸叫抑制的可行性，提出基于U-Net的频域掩膜估计方法，并通过设计多个从基础到高级的模型变体，系统性地研究注意力机制、残差连接、空洞卷积和生成对抗网络等先进技术对啸叫抑制效果的影响，为深度学习在声学信号处理领域的应用提供新的理论参考。在实践层面，本研究成果有望为新一代智能扩声系统的开发提供技术支撑，推动啸叫抑制技术从基于手工特征和规则的传统方法向基于数据驱动的智能方法转变，从而在教室、会议室、演出场馆等实际应用场景中提升用户的听觉体验，保护音频设备的安全运行。

## 1.2 国内外研究现状

### 1.2.1 传统啸叫抑制方法

声反馈啸叫问题自扩声系统诞生以来就伴随着工程实践，其研究历史可以追溯到上世纪中叶。早期的啸叫抑制主要依赖人工干预手段[3]，由专业音频工程师根据经验通过手动调整均衡器来降低啸叫频点的增益，或通过调整麦克风和扬声器的空间布局（如增大两者距离、使用定向麦克风等）来减少声反馈的发生概率。这些方法对操作人员的专业技能要求较高，且难以实时应对动态变化的声学环境。随着数字信号处理技术的发展，自动化算法逐渐取代了人工操作，形成了多种基于DSP的啸叫抑制方案。

**移频/移相法**是最早出现的自动化啸叫抑制方法之一[5]。该方法由Berdahl和Harris等人在2010年前后进行了系统性的研究与改进。其基本原理是通过在信号传输链路中引入一个频率偏移量Δf（通常为3~8Hz），使得反馈信号在每次环路循环中发生频率偏移，从而破坏啸叫产生的相位条件。移相法则是通过对信号施加时变的相位调制来达到类似的效果。从信号处理的角度来看，移频操作等效于对信号进行了频域的平移变换，这种变换虽然能有效抑制啸叫，但同时也会导致信号频谱结构的改变，在听觉上表现为音质的明显劣化。因此，移频法主要适用于对音质要求相对较低的纯语音扩声场景，在音乐演出等对音质要求较高的场合则较少采用。

**陷波抑制法（NHS）**是目前应用最为广泛的啸叫抑制方案之一[6][7]。该方法的核心在于两个关键步骤：啸叫频率点检测和窄带陷波滤波。啸叫检测算法通常基于频谱分析，通过提取信号的频域特征（如频谱峰值、频率稳定性、振铃时间等）来判断是否存在啸叫。van Waterschoot和Moonen在2010年对多种基于陷波滤波器的啸叫抑制方案中的检测准则进行了系统性的对比评估[7]，揭示了不同检测标准在准确性和鲁棒性方面的差异。在检测到啸叫频率点后，系统会在该频率处自动生成一个Q值较高的陷波滤波器（通常带宽为5~30Hz）进行频率点抑制。NHS方法的优势在于处理策略直观、计算效率较高，但其性能瓶颈在于：啸叫频率的误检会导致正常语音成分被错误衰减，而漏检则无法及时消除啸叫；当系统增益较高时，可能同时出现多个啸叫频率点，需要部署大量陷波器，这将严重破坏信号的频谱完整性。

**自适应反馈消除法（AFC）**是目前理论最为完备的啸叫抑制技术[8][9]。该方法的核心思想是利用自适应滤波器在线估计扬声器到麦克风之间的声反馈传递函数，并从麦克风信号中减去估计得到的反馈分量。早期的研究主要采用最小均方（LMS）算法和归一化最小均方（NLMS）算法来更新自适应滤波器的系数。随着研究的深入，改进的变步长LMS算法、仿射投影算法（APA）和基于子带分解的多速率自适应算法等被相继提出，以提高收敛速度和降低稳态误差。然而，AFC方法在实际应用中面临一个核心难题：目标信号与反馈信号均来源于同一声源，两者之间的高度相关性会导致自适应滤波器产生估计偏差。为了缓解这一问题，研究者们提出了多种去相关预处理技术，包括前向预测去相关、非线性函数变换（如半波整流）以及注入随机探测噪声等[8]。但这些去相关手段在改善自适应滤波性能的同时，也不可避免地引入了额外的信号失真，从而在整体上降低了语音质量。Wang等人于2020年提出了一种将预测误差方法与AFC相结合的方案，在列车公共广播系统中实现了啸叫抑制[9]，但该方法仍难以完全消除去相关处理对音质的影响。

综合来看，传统啸叫抑制方法虽然在工程应用中发挥了重要作用，但均存在一个共性不足：在抑制啸叫的同时难以有效保持原始信号的质量。移频法破坏了信号的频率结构，陷波法引入了频谱失真，自适应反馈消除法则受限于信号相关性问题。这些局限性从根本上源于传统方法需要对声反馈过程进行显式的数学建模或规则设计，而实际声学环境中的反馈过程涉及复杂的房间声学特性、多次反射叠加和非线性失真等因素，难以用简洁的数学模型精确描述。这一困境为深度学习技术的引入提供了契机。

### 1.2.2 深度学习在音频处理中的应用

近年来，深度学习技术的飞速发展为音频信号处理领域带来了革命性的变化，在多个子领域取得了超越传统方法的成果。以下从与本课题密切相关的几个方向进行综述。

**语音增强**是深度学习在音频处理中最早取得突破性进展的领域之一。语音增强的目标是从带噪语音中去除噪声干扰，恢复干净的语音信号。早期的研究主要采用基于频域掩膜的方法，如Wang和Chen等人提出的基于深度神经网络（DNN）的语音增强方法[16]，通过学习带噪语音时频表示到理想掩膜（Ideal Binary/Ratio Mask）的映射来实现噪声抑制。随后，卷积神经网络（CNN）和循环神经网络（RNN）等更强大的网络架构被引入，如Tan和Wang提出的基于卷积循环网络（CRN）的语音增强系统[17]，在保持较低计算复杂度的同时取得了优异的增强效果。近年来，以Demucs[18]、FullSubNet[19]等为代表的端到端时域和时频域联合处理模型进一步推动了语音增强技术的发展。这些研究成果表明，深度学习方法能够从数据中自动学习复杂的噪声模式，并有效地将语音信号与噪声分离，在客观评价指标和主观听觉质量上均显著优于传统的谱减法、维纳滤波等方法。

**语音分离**技术与语音增强有着密切的关联，其目标是从多个说话人的混合语音中分离出各个独立的说话人信号。深度聚类（Deep Clustering）[20]、排列不变训练（PIT）[21]和时域分离网络（TasNet）[22]等方法的提出极大地推动了该领域的发展。语音分离任务与啸叫抑制任务在形式上具有相似性：两者都是从混合信号中提取目标信号。不同的是，语音分离中需要分离的是多个说话人的语音，而啸叫抑制中需要分离的是目标语音和声反馈引起的啸叫分量。正是这种任务形式上的相似性，使得语音分离领域的许多技术方案可以迁移到啸叫抑制问题中。

**U-Net在音频频谱处理中的应用**是本课题的重要技术背景。U-Net最初由Ronneberger等人于2015年为生物医学图像分割任务而提出[13]，其编码器-解码器结构配合跳跃连接的设计在需要精确定位的像素级预测任务中表现出色。由于音频信号的时频表示（如STFT幅度谱）在数据结构上与二维图像类似，研究者们自然地将U-Net架构迁移到了音频处理领域。在音乐源分离任务中，Jansson等人使用U-Net架构从混合音乐频谱中分离出人声和伴奏[14]，取得了优异的效果。在语音增强领域，以U-Net为骨干网络的多种变体也被广泛研究和应用[15]。此外，针对U-Net的改进方案不断涌现，如在跳跃连接中引入注意力机制[23]使网络自动聚焦于关键特征区域，在编码器中引入残差连接[24]缓解深层网络的梯度消失问题，以及在瓶颈层使用空洞卷积[25]扩大感受野以捕获更丰富的上下文信息。这些改进技术的提出为构建更强大的音频处理模型提供了丰富的技术储备。

**深度学习啸叫抑制的探索现状**方面，由于啸叫抑制相较于语音增强和语音分离是一个更为细分的领域，目前利用深度学习技术进行啸叫抑制的研究工作还相对较少，但近两年已开始出现具有开创性的探索。Zhang等人于2023年在ICASSP会议上发表了Deep AHS[26]，首次提出了将声学啸叫抑制问题建模为深度学习框架下的监督学习问题的思路，通过神经网络从麦克风信号中直接估计目标语音，无需进行显式的啸叫检测，验证了深度学习方法应用于啸叫抑制的可行性。同年，该团队在ASRU会议上进一步提出了联合声学回声与声学啸叫抑制的深度学习方案[27]，将AEC和AHS任务统一在同一框架下进行处理，展示了深度学习在处理多种声学退化问题上的灵活性。然而，上述工作在网络架构设计、损失函数优化、数据增强策略等方面仍有较大的改进空间，特别是在如何充分利用U-Net架构的多尺度特征提取能力以及注意力机制等先进技术方面，尚缺乏系统性的研究。

综合以上分析，深度学习技术在音频处理领域的成功应用为啸叫抑制提供了新的技术路径和研究思路。然而，目前将深度学习应用于啸叫抑制的研究仍处于初步阶段，特别是在基于U-Net架构的频域掩膜估计方法、多种先进网络技术的系统集成以及与传统方法的系统性对比等方面，尚缺乏深入的研究。本研究正是基于这一背景，旨在填补上述研究空白，推动深度学习技术在啸叫抑制领域的深入应用。

## 1.3 研究内容与结构安排

基于以上对传统啸叫抑制方法局限性和深度学习技术优势的分析，本文提出了一种基于深度U-Net的声反馈啸叫抑制方法。本文的核心研究思路是：将啸叫抑制问题抽象为频域掩膜估计任务，利用U-Net卷积神经网络学习带啸叫频谱到干净频谱之间的非线性映射关系，通过端到端的方式实现啸叫的自动抑制。

本文的主要工作包括以下几个方面：

（1）**建立了声反馈啸叫的理论模型和仿真数据生成框架。** 基于单通道闭环声学增益系统模型，利用房间脉冲响应（RIR）生成器模拟声反馈路径，通过设置随机化的增益、延迟和房间声学参数，构建了大规模的干净语音-带啸叫语音配对数据集，为深度学习模型的训练提供了数据基础。

（2）**设计了从基础到高级的多个U-Net模型变体。** 逐步探索不同网络设计策略对啸叫抑制效果的影响，包括：3层基线U-Net（AudioUNet3）作为最小可行方案；5层标准U-Net（AudioUNet5）扩大网络容量；注意力U-Net（AudioUNet5Attention）引入注意力门机制聚焦啸叫相关频率区域；综合优化U-Net（AudioUNet5Optimized）集成注意力门、残差连接和空洞卷积三重改进；GAN增强U-Net（AudioUNet5GAN）引入生成对抗训练策略提升增强质量。

（3）**实现了三种传统啸叫抑制方法作为基线对比。** 包括移频法、增益抑制法和自适应反馈消除法，为全面评估深度学习方法的优势提供了参考基准。

（4）**从多个维度进行了系统的实验对比分析。** 包括模型架构对比、消融实验、损失函数对比、训练策略对比和数据增强对比等实验，全面评估了各技术组件的贡献和模型的综合性能。

本文的结构安排如下：

**第1章 绪论。** 介绍了研究的背景与意义，综述了传统啸叫抑制方法和深度学习在音频处理领域的研究现状，概述了本文的研究内容和论文结构。

**第2章 相关理论基础。** 介绍本研究所涉及的理论基础，包括声学啸叫的数学模型、U-Net卷积神经网络的架构原理、注意力机制、残差连接与空洞卷积、生成对抗网络以及短时傅里叶变换等关键技术，为后续章节的实验设计提供理论支撑。

**第3章 实验设计。** 详细阐述实验的整体设计方案，包括数据集的构建方法、特征提取的具体流程、五个U-Net模型变体的网络结构设计、三种传统方法的实现方案、损失函数的设计、数据增强策略以及评价指标的定义。

**第4章 实验组织。** 描述实验的具体实施过程，包括实验环境的搭建、六组实验的详细方案（模型训练对比、统一评估、消融实验、损失函数对比、训练策略对比、数据增强对比）以及结果可视化方案。

**第5章 实验结果与分析。** 呈现和分析各项实验的结果，包括传统方法的基线结果、五个U-Net模型变体的性能演进对比、消融实验中各组件的贡献分析，以及损失函数、训练策略和数据增强的对比分析，并结合频谱可视化图对啸叫抑制效果进行直观展示。

**第6章 结语。** 总结本文的主要工作和研究结论，分析当前研究中存在的不足，并对未来的研究方向进行展望。

---

## 参考文献

[1] 周璐. 影响自适应反馈抵消啸叫抑制算法性能的声学因素分析[D]. 南京大学, 2012.
[2] 杨阳. 声学系统中反馈抑制器的研究和DSP实现[D]. 广州大学, 2016.
[3] 赵明. 啸叫检测与抑制扩声系统设计[D]. 西北大学, 2018.
[4] 王凤森. 扩声系统中啸叫抑制算法的研究[D]. 五邑大学, 2020.
[5] Berdahl E, Harris D. Frequency shifting for acoustic howling suppression[C]. Proceedings of the 13th International Conference on Digital Audio Effects, Graz, Austria, 2010.
[6] Loetwassana W, Punchalard R, Lorsawatsiri A, et al. Adaptive howling suppressor in an audio amplifier system[C]. Asia-Pacific Conference on Communications, IEEE, 2007: 445-448.
[7] van Waterschoot T, Moonen M. Comparative evaluation of howling detection criteria in notch-filter-based howling suppression[J]. Journal of the Audio Engineering Society, 2010, 58(11): 923-940.
[8] 甘华国. 基于深度学习的音频啸叫处理方法研究[D]. 广州大学, 2021.
[9] Wang G, Liu Q, Wang W. Adaptive feedback cancellation with prediction error method and howling suppression in train public address system[J]. Signal Processing, 2020, 167: 107279.
[10] LeCun Y, Bengio Y, Hinton G. Deep learning[J]. Nature, 2015, 521(7553): 436-444.
[11] Zhang H, Tan K, Wang D L. Deep learning for joint acoustic echo and noise cancellation with nonlinear distortions[C]. INTERSPEECH, 2019: 4255-4259.
[12] Zhang S, Kong Y, Lv S, et al. FT-LSTM based complex network for joint acoustic echo cancellation and speech enhancement[J]. arXiv preprint arXiv:2106.07577, 2021.
[13] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]. MICCAI, 2015: 234-241.
[14] Jansson A, Humphrey E, Montecchio N, et al. Singing voice separation with deep U-Net convolutional networks[C]. International Society for Music Information Retrieval Conference, 2017.
[15] Cho Y, Kim H. Speech enhancement using U-Net with spectral magnitude approximation[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.
[16] Wang Y, Wang D L. A deep neural network for time-domain signal reconstruction[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015: 4390-4394.
[17] Tan K, Wang D L. A convolutional recurrent neural network for real-time speech enhancement[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019.
[18] Défossez A, Usunier N, Bottou L, et al. Demucs: Deep extractors for music sources[C]. International Society for Music Information Retrieval Conference, 2019.
[19] Hao X, Su X, Horaud R, et al. FullSubNet: A full-band and sub-band fusion model for real-time single-channel speech enhancement[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.
[20] Hershey J R, Chen Z, Le Roux J, et al. Deep clustering: Discriminative embeddings for segmentation and separation[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2016.
[21] Yu D, Kolbaek M, Tan Z H, et al. Permutation invariant training of deep models for speaker-independent multi-talker speech separation[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017.
[22] Luo Y, Mesgarani N. TasNet: Surpassing ideal time-frequency magnitude masking for speech separation[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019, 27(10): 1495-1508.
[23] Oktay O, Schlemper J, Folgoc L L, et al. Attention U-Net: Learning where to look for the pancreas[C]. Medical Imaging with Deep Learning, 2018.
[24] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016: 770-778.
[25] Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.
[26] Zhang H, Yu M, Yu D. Deep AHS: A deep learning approach to acoustic howling suppression[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023: 1-5.
[27] Zhang H, Yu M, Yu D. Deep learning for joint acoustic echo and acoustic howling suppression in hybrid meetings[C]. IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2023: 1-7.