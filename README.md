# Hindsight-Experience-Replay---Bit-Flipping
### Simple bit flipping with sparse rewards using HER, similarly to the original paper.

This project contains a simple implementation of the bit flipping experiment from the paper "Hindsight Experience Replay" [1] (HER) by OpenAI researchers. In this experiment an agent is given an initial binary state vector and a binary goal vector, and must get from the initial state to the goal state by flipping a bit at each step. The agent is given -1 reward for every step that the goal is not reached and 0 when it reaches it, making it a sparse reward problem. using HER, the agent can gradualy increase its reachable set and eventually arrive at the intented goals.

For the 15 bit experiment, after 5000 epsidoes the agent can get to the goal ~90% of the time:
![Alt text](https://user-images.githubusercontent.com/46422351/53296086-14701500-3811-11e9-8281-6a9f513c7764.png)

During training, the minimum distance to the goal during each episode can be seen to gradually decrease:
![Alt text](https://user-images.githubusercontent.com/46422351/53296076-f30f2900-3810-11e9-8be5-ecb3bfb8abdd.png)

I have written a [Medium post](https://towardsdatascience.com/reinforcement-learning-with-hindsight-experience-replay-1fee5704f2f8) explaining the intuition behind Hindsight Experience Replay, feel free to check it out.


1. Andrychowicz, Marcin, et al. "Hindsight experience replay." Advances in Neural Information Processing Systems. 2017.


