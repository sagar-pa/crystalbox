from gymnasium import register


register(
    id='cc-rl-v0',
    entry_point='cc_rl.network_sim:SimulatedNetworkEnv')