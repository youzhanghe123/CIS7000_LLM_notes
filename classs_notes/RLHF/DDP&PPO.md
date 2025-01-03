1. DPO与PPO的核心区别
   
DPO (Direct Preference Optimization)

目标：基于人类偏好数据，直接优化模型的行为，使其更符合偏好。
应用场景：通常用于 语言模型优化，如通过人类反馈提升生成文本的质量（比如 ChatGPT 的 RLHF 阶段）。
特点：DPO 不依赖明确的强化学习奖励函数，而是直接通过比较生成的样本与人类偏好样本之间的相对优劣来更新模型。

PPO (Proximal Policy Optimization)

目标：在不让策略发生剧烈变化的情况下，最大化预定义的奖励函数。
应用场景：更常见于经典的强化学习任务，比如游戏控制、机器人操作等。
特点：PPO 引入了「裁剪」机制，限制每次策略更新的步长，防止策略变化太大而导致性能下降。

2. 优化方式
   
DPO

假设有一组模型输出：一个被偏好（例如：人类标注为「更好」），另一个不被偏好，
优化目标是提升被偏好样本的生成概率，降低不被偏好样本的生成概率。
使用的损失函数类似于对比学习 （参见loss_function_pair_wise.png). DPO不需要训练额外的reward model，因为DPO直接优化生成更好答案的概率。

PPO

直接更新策略，目标是最大化策略与旧策略相比的收益，同时不让新策略过多背离就策略，损失函数参见（RLHF_common_objective.png).
PPO需要训练额外的reward model，来提供强化学习中的r(t)。
加入裁剪机制，确保新策略与旧策略的偏离度在一定范围内。
