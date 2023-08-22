# Melting Pot Experiments

Attempt to replicate some of the research work in Multi-Agent Reinforcement Learning inside the [Melting Pot](https://github.com/deepmind/meltingpot) testing suite

Progresss so far:

- PPO+LSTM with Classifier Norm Model (CNM) on Melting Pot 1.0
- 4 social outcome metrics

see my [Weights and Biases report](https://wandb.ai/marshin/Meltingpot/reports/Meltingpot-initial-trials--VmlldzoyNTIxMDg5) for more details

Current progres:

- Replicate Model Other Agent (MOA) from [3] on Melting Pot 2.0

## Relevant papers

1. Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017). [Multi-agent reinforcement learning in sequential social dilemmas](https://arxiv.org/abs/1702.03037). In Proceedings of the 16th Conference on Autonomous Agents and MultiAgent Systems (pp. 464-473).

2. Hughes, E., Leibo, J. Z., Phillips, M., Tuyls, K., Dueñez-Guzman, E., Castañeda, A. G., Dunning, I., Zhu, T., McKee, K., Koster, R., Tina Zhu, Roff, H., Graepel, T. (2018). [Inequity aversion improves cooperation in intertemporal social dilemmas](https://arxiv.org/abs/1803.08884). In Advances in Neural Information Processing Systems (pp. 3330-3340).

3. Jaques, N., Lazaridou, A., Hughes, E., Gulcehre, C., Ortega, P. A., Strouse, D. J., Leibo, J. Z. & de Freitas, N. (2018). [Intrinsic Social Motivation via Causal Influence in Multi-Agent RL](https://arxiv.org/abs/1810.08647). arXiv preprint arXiv:1810.08647.

4. Vinitsky, E., Köster, R., Agapiou, J. P., Duéñez-Guzmán, E., Vezhnevets, A. S., & Leibo, J. Z. (2021). [A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings](https://arxiv.org/abs/2106.09012). arXiv preprint arXiv:2106.09012.

## Changelog

changes on the external libraries outside of my experiment code are documented here to get the project running.

### 2023-08-22

- Exposed `AVATAR_IDS_IN_VIEW` which is a debug obs in the current Melting Pot 2.0. This is neccessary info for MOA agent visiblity for influence calculations.
  - Substrates modified are 'allelopathic_harvest\_\_open' and ‘clean_up'
    ```python
        avatar_object["components"].append({
            "component": "AvatarIdsInViewObservation",
        })
    ```
- Modify `shimmy` to call `from meltingpot import substrate` directly
