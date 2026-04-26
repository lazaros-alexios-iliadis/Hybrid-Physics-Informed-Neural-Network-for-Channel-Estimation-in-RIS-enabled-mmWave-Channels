# Hybrid Physics-Informed Neural Network for Channel Estimation in RIS-enabled mmWave Channels
This project is the code for the paper Hybrid Physics-Informed Neural Network for Channel Estimation in RIS-enabled mmWave Channels, which was presented in EuCAP 2026.

## Abstract
Reconfigurable intelligent surface (RIS)-enabled millimeter wave (mmWave) communication requires accurate channel estimation for optimal beamforming and phase
control. Model-based methods face high computational complexity, while traditional data-driven approaches fail to exploit the underlying physical structure. This 
work applies a hybrid physics-informed neural network (PINN) which utilizes physics-based preprocessing as augmented input features while maintaining direct 
end-to-end learning of the cascaded channel matrix. We concatenate raw observations with physics-motivated transformations obtained through combining RIS phase 
matrices, leveraging both data patterns and electromagnetic principles. The proposed DL architecture consists of four fully-connected layers with layer normalization
and dropout for robust generalization. Extensive simulations on RIS-enabled mmWave SIMO system demonstrate that the hybrid PINN achieves a satisfactory normalized 
mean square error (NMSE), outperforming least squares, multilayer perceptron, and model-based unfolding networks (UNF). Furthermore, the proposed method exhibits 
computational efficiency and low inference time, making it suitable for real-time deployment in next-generation wireless systems.

## Main reference
This work builds upon the following paper:
J. He, H. Wymeersch, M. Di Renzo and M. Juntti, "Learning to Estimate RIS-Aided mmWave Channels," in IEEE Wireless Communications Letters, vol. 11, no. 4, 
pp. 841-845, April 2022, doi: 10.1109/LWC.2022.3147250.

The authors provided their code on the following GitHub repo:
https://github.com/jiguanghe/RISCE

## Cite
The correct citation will be added after the publication of EuCAP's proceedings.
