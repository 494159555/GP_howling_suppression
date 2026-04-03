# 第6章 结语

本文将深度学习技术引入声学啸叫抑制领域，提出了一种基于U-Net卷积神经网络的频域掩膜估计方法，通过学习带啸叫频谱到干净频谱之间的非线性映射关系，实现端到端的啸叫抑制。本文设计了从基础到高级的多个U-Net模型变体，系统性地探索了注意力机制、残差连接、空洞卷积和生成对抗网络等先进技术对啸叫抑制效果的影响，同时从损失函数、训练策略和数据增强等多个维度进行了全面的实验优化。经过理论分析、仿真建模、模型训练和系统实验，验证了所提方法的有效性和深度学习技术应用于啸叫抑制的可行性。

## 6.1 工作总结

本文的主要工作和贡献可以归纳为以下几个方面：

（1）**建立了声反馈啸叫的理论模型和仿真数据生成框架。** 本文基于单通道闭环声学增益系统，建立了啸叫产生的数学模型，阐明了啸叫形成的振幅条件和相位条件。在此基础上，利用房间脉冲响应（RIR）生成器模拟声反馈路径，通过设置随机化的增益（5~10dB）、延迟（0.02~0.2秒）和房间声学参数（混响时间RT60为0.1~0.5秒），构建了大规模的干净语音-带啸叫语音配对数据集。该仿真框架忠实地再现了声反馈的递归信号叠加过程，为深度学习模型的训练提供了可靠的数据基础。

（2）**设计了从基础到高级的五个U-Net模型变体，逐步探索不同网络设计策略的影响。** 本文按照从简单到复杂的递进思路设计了五种模型：3层基线U-Net（AudioUNet3，参数量1.2M）验证了U-Net架构应用于啸叫抑制的可行性；5层标准U-Net（AudioUNet5，参数量4.5M）通过增加网络深度显著提升了性能；注意力U-Net（AudioUNet5Attention，参数量4.8M）在跳跃连接中引入注意力门机制，使网络能够自动聚焦于啸叫相关的频率区域；综合优化U-Net（AudioUNet5Optimized，参数量6.2M）集成了注意力门、残差连接和空洞卷积三重改进，在SI-SDR（11.45dB）和STOI（0.92）上均取得了最优表现；GAN增强U-Net（AudioUNet5GAN）通过生成对抗训练策略，在PESQ指标上达到了所有方法中最高的3.05，接近"良好"的语音质量水平。

（3）**实现了三种传统啸叫抑制方法作为基线对比。** 包括移频法、增益抑制法和自适应反馈消除法，为全面评估深度学习方法的优势提供了参考基准。实验结果表明，深度学习方法在所有评价指标上均大幅超越了传统方法：最优深度学习模型的SI-SDR（11.45dB）相比最佳传统方法自适应反馈消除法（-0.72dB）提升了12.17dB，验证了深度学习方法在啸叫抑制领域的显著优势。

（4）**从多个维度进行了系统的实验对比分析。** 本文组织了六组实验，涵盖模型架构对比、消融实验、损失函数对比、训练策略对比和数据增强对比等方面。消融实验量化了各组件的贡献度：残差连接贡献最大（移除后SI-SDR下降1.60dB），注意力机制次之（下降1.33dB），空洞卷积最小（下降0.92dB），且三个组件之间存在正向协同效应。在训练优化方面，复合损失函数（多分辨率STFT损失+SI-SDR损失）和Warmup+CosineDecay学习率调度策略被确定为最优训练配置，综合数据增强策略进一步将SI-SDR提升至12.05dB，PESQ突破3.0。

综合以上研究，本文得出以下主要结论：

- **深度学习方法的显著优越性。** 基于U-Net的深度学习方法通过数据驱动的方式学习复杂的非线性映射关系，从根本上避免了传统方法面临的啸叫检测不准确、信号相关性干扰和音质损失等问题，在信号重建质量、主观听觉质量和语音可懂度等维度上均大幅超越了传统方法。
- **网络架构设计的有效性。** 注意力机制、残差连接和空洞卷积三项技术改进均对啸叫抑制效果有正向贡献，分别从特征选择精度、特征学习效率和上下文感知范围三个互补的维度提升了模型性能。其中残差连接通过改善梯度传播效率对模型性能的贡献最大。
- **训练优化策略的重要性。** 合理的损失函数设计、学习率调度策略和数据增强策略能够进一步提升模型性能，特别是多源监督信号的复合损失函数和多维度数据增强策略对模型的泛化能力有显著提升作用。

## 6.2 不足与展望

尽管本文提出的方法在客观评价指标上取得了较好的结果，但仍存在一些不足之处，需要进一步研究改进：

**（1）仿真数据与真实场景的差距。** 本文的实验数据全部基于仿真生成，虽然仿真框架考虑了房间声学参数的随机化和非线性失真等因素，但与真实声学环境中的啸叫现象仍存在一定差距。真实场景中的声反馈路径可能包含更复杂的非线性特性、时变特性和多径效应，麦克风和扬声器的频率响应特性也更为多样。未来工作应采集真实环境下的啸叫数据，或利用迁移学习技术将仿真数据训练的模型适配到真实场景中，以提升方法的实际应用能力。

**（2）未进行主观听音测试。** 本文的评估主要依赖于SI-SDR、PESQ和STOI等客观评价指标，虽然这些指标能够在一定程度上反映音频质量，但主观听觉感受才是评判啸叫抑制效果的最终标准。未来工作应组织规模化的主观听音测试（如MOS评分），从听觉感知的角度全面评估各类方法的实际效果，并分析客观指标与主观评价之间的相关性。

**（3）实时在线处理的实现。** 本文实验中模型推理时间在8.5ms至22.7ms之间，虽然在一定程度上满足了实时处理的要求，但尚未实现完整的在线流式处理系统。实际扩声系统要求极低的端到端延迟（通常小于20ms），未来需要探索模型压缩、知识蒸馏和量化推理等技术，降低模型的计算延迟和资源占用，实现真正的实时在线啸叫抑制。

**（4）端到端时域处理方法的探索。** 本文采用基于STFT的频域处理范式，虽然这一范式在音频处理中被广泛使用，但STFT/iSTFT操作不可避免地引入了额外的时间延迟和信息损失。近年来，以Conv-TasNet、Demucs等为代表的端到端时域处理方法在语音分离和语音增强任务中取得了优异的效果。未来可以探索将端到端时域处理方法应用于啸叫抑制任务，省去STFT/iSTFT的中间环节，有望进一步降低处理延迟并提升信号重建质量。

**（5）更轻量化的网络结构设计。** 本文最优模型AudioUNet5Optimized的参数量为6.2M，虽然可以通过GPU进行高效推理，但在资源受限的嵌入式设备上部署仍面临挑战。未来可以探索轻量化网络设计技术，如深度可分离卷积、神经网络架构搜索（NAS）和模型剪枝等，在保持性能的前提下大幅减少参数量和计算量，使模型能够部署在DSP芯片或移动端设备上。

**（6）多通道和多人场景的扩展。** 本文的研究聚焦于单通道啸叫抑制问题，而实际扩声系统通常是多麦克风多扬声器的配置。多通道场景下，不同通道的声反馈路径和啸叫特性各不相同，为啸叫抑制带来了更大的挑战。未来可以将单通道方法扩展到多通道场景，利用多通道的空间信息进行联合处理。此外，在多人同时发言的会议场景中，啸叫抑制与语音分离、声学回声消除等任务的联合优化也是一个值得探索的方向。

---

## 参考文献

[1] Zhang H, Yu M, Yu D. Deep AHS: A deep learning approach to acoustic howling suppression[C]. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023: 1-5.

[2] Zhang H, Yu M, Yu D. Deep learning for joint acoustic echo and acoustic howling suppression in hybrid meetings[C]. IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2023: 1-7.

[3] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]. International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015: 234-241.

[4] Défossez A, Usunier N, Bottou L, et al. Demucs: Deep extractors for music sources[C]. International Society for Music Information Retrieval Conference, 2019.

[5] Luo Y, Mesgarani N. Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation[J]. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2019, 27(10): 1495-1508.

[6] Oktay O, Schlemper J, Folgoc L L, et al. Attention U-Net: Learning where to look for the pancreas[C]. Medical Imaging with Deep Learning, 2018.

[7] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016: 770-778.

[8] Chen L C, Papandreou G, Schroff F, et al. Rethinking atrous convolution for semantic image segmentation[J]. arXiv preprint arXiv:1706.05587, 2017.

[9] Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[C]. Advances in Neural Information Processing Systems (NeurIPS), 2014.