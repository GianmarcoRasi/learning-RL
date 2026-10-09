def transition(state, action, size, step_reward=-1.0, goal_reward=10.0):
    """Return the deterministic outcome of a nonterminal LineWorld action."""
    if size < 2:
        raise ValueError("size must be at least 2")
    if not 0 <= state < size - 1:
        raise ValueError("state must be a nonterminal position")
    if action not in (0, 1):
        raise ValueError("Action must be 0 (left) or 1 (right)")
    next_state = max(0, state - 1) if action == 0 else state + 1
    terminated = next_state == size - 1
    return next_state, goal_reward if terminated else step_reward, terminated


class LineWorld:
    def __init__(self, size=5, step_reward=-1.0, goal_reward=10.0):
        if size < 2:
            raise ValueError("size must be at least 2")
        self.size, self.goal = size, size - 1
        self.step_reward, self.goal_reward = step_reward, goal_reward
        self.state = 0

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state == self.goal:
            raise RuntimeError("Episode terminated: call reset()")
        if action not in (0, 1):
            raise ValueError("Action must be 0 (left) or 1 (right)")
        self.state, reward, terminated = transition(
            self.state, action, self.size, self.step_reward, self.goal_reward)
        return self.state, reward, terminated


