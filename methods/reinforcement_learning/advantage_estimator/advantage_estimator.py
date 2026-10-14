class OneStepAdvantageEstimator():
    def __init__(self, discount_factor):
        self.discount_factor = discount_factor
    def __call__(self, reward, value_function_next, value_function_now):
        return reward + self.discount_factor * value_function_next - value_function_now


class NStepAdvantageEstimator:
    def __init__(self, discount_factor, n_step):
        self.discount_factor = discount_factor
        self.n_step = n_step
        self.td_error = []

    def __call__(self, reward, value_function_next, value_function_now, terminated, truncated):
        if terminated: value_function_next = 0
        self.td_error.append(reward + self.discount_factor * value_function_next - value_function_now)
        if not (terminated or truncated):
            return None
        advantages = []
        T = len(self.td_error)
        for t in range(T):
            advantage = 0
            for k in range(self.n_step):
                if t + k >= T: break
                advantage += self.discount_factor ** k * self.td_error[t + k]
            advantages.append(advantage)
        self.td_error.clear()
        return advantages



class GeneralizedAdvantageEstimator():
    def __init__(self, discount_factor, trace_decay):
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        self.td_error = []
    def __call__(self, reward, value_function_next, value_function_now, terminated, truncated):
        # value function of a termination state should be 0 
        if terminated: value_function_next = 0
        self.td_error.append(reward + self.discount_factor * value_function_next - value_function_now)
        if not (terminated or truncated):
            return None
        advantages = []
        advantage = 0
        for td_error in reversed(self.td_error):
            advantage = advantage * self.discount_factor * self.trace_decay + td_error
            advantages.append(advantage)
        self.td_error.clear()
        return list(reversed(advantages))
        
OneStepAE = OneStepAdvantageEstimator
NStepAE = NStepAdvantageEstimator
GAE = GeneralizedAdvantageEstimator