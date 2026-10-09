# Project glossary

This glossary collects the vocabulary introduced in Lessons 1–4, including the
hands-on labs and progressive exercises. Definitions follow the lessons; each
section links to the lesson that explains them. Use these terms consistently in
lessons, source code, command help, output, and plots. Extend the glossary when a
new lesson defines a term. Familiar examples such as a courier's position are
concrete instances of a **state**, rather than new technical terms.

## Foundations

Source: [Lesson 1](../lessons/RL-lesson-01-rl-foundations-01.md) and its
[hands-on tutorial](../lessons-hands-on/RL-lesson-01-rl-foundations-02-hands-on.md).

- **Reinforcement learning (RL):** Learning what actions to take through experience when choices have consequences over time.
- **Supervised learning:** Learning from examples that include target answers; RL feedback instead scores outcomes without labeling the best action.
- **Agent:** The decision maker that chooses actions.
- **Environment:** The system that responds to actions with rewards and next states.
- **Sequential decision making:** Decisions affect the situations in which later decisions occur.
- **State**, $S_t$ or $s$: The current situation described by the variables included in the model. In LineWorld it is the numbered position.
- **Action**, $A_t$ or $a$: A choice available to the agent in a state. LineWorld codes left as `0` and right as `1`.
- **Next state**, $S_{t+1}$ or $s'$: The state produced after the current action.
- **Reward**, $R_{t+1}$ or $r$: The numerical score of one transition. It is distinct from a return or value.
- **Decision step**, $t$: The index at which the agent chooses an action. A completed interaction step is one transition, even when the state does not change.
- **Transition:** One interaction, recorded as $(S_t,A_t,R_{t+1},S_{t+1})$.
- **Trajectory**, $\tau$: A sequence of connected transitions; it may be a segment, a complete episode, or an ongoing sequence.
- **Episode:** An interaction sequence with a task-defined end. The runner can also record a truncated episode before that end.
- **Episodic task:** A task with episodes that end on entering a terminal state.
- **Continuing task:** A task with no natural end.
- **Terminal state:** A state that ends the episode; no further actions or rewards follow. Any completion reward belongs to the transition entering it.
- **Termination:** Reaching the task-defined end of an episode.
- **Truncation / cutoff / external time limit:** Stopping a record while the underlying task would continue. Missing future rewards must not be confused with zero future reward.
- **Task deadline / finite horizon:** A task-defined time boundary. Include time remaining in the state when it affects outcomes or decisions.
- **Policy**, $\pi$: A rule for choosing actions in each state; $\pi(a\mid s)$ is the probability of choosing $a$ given $s$.
- **Deterministic policy:** A policy specifying one action in each state.
- **Stochastic policy:** A policy specifying action probabilities.
- **Stationary policy:** A policy whose action probabilities depend on state, without changing with decision time.
- **Fixed policy:** A supplied policy that is not updated by learning in the experiment. Both Lesson 1 policies are fixed.
- **Uniform random policy:** Chooses each available action with equal probability.
- **Markov Decision Process (MDP):** A model specifying states, available actions, transition probabilities, and reward generation, with the Markov property. The lessons list $(\mathcal S,\mathcal A,P,R,\gamma)$.
- **State space**, $\mathcal S$: The set of possible states.
- **Action space**, $\mathcal A$: The set of actions; availability may depend on the state.
- **Transition model**, $P(s'\mid s,a)$: The probability of the next state given the current state and action. The joint model $p(s',r\mid s,a)$ also specifies the reward.
- **Markov property:** Current state and action summarize all history needed to predict the distribution of the next state and reward. This does not imply deterministic outcomes.
- **Observation:** Information the agent receives; it may omit variables needed for a full Markov state.
- **Return**, $G_t$: The sum of future rewards, weighted by powers of the discount factor: $G_t=\sum_{k\geq0}\gamma^kR_{t+k+1}$, ending at termination for a finite episode.
- **Discount factor**, $\gamma$: The weight controlling how strongly future rewards count. The next reward always has weight one; $\gamma=1$ gives an undiscounted sum.
- **Expected return:** A probability-weighted average of returns over possible futures.
- **Reward sum / undiscounted return:** The sum of rewards with no discounting; it equals the return at $\gamma=1$ over the same recorded trajectory.
- **Recorded return:** The discounted sum over recorded transitions only. At truncation, it excludes unobserved continuation.
- **Cumulative discounted reward**, $C_n$: Rewards accumulated from the start through $n$ recorded transitions, $\sum_{t=0}^{n-1}\gamma^tR_{t+1}$. An intermediate $C_n$ is not the future return $G_n$.
- **Reward design:** Choosing the scoring rule the agent will optimize.
- **Sparse rewards:** Rewards given only at selected events; the lab uses zero for ordinary transitions and one for reaching the goal.
- **LineWorld:** The shared deterministic environment with a line of numbered states, a reflecting left boundary, and a terminal goal at the right boundary.

## Values and policy evaluation

Source: [Lesson 2](../lessons/RL-lesson-02-value-functions-bellman-01.md) and its
[hands-on tutorial](../lessons-hands-on/RL-lesson-02-value-functions-bellman-02-hands-on.md).

- **Value:** Expected return across possible futures, distinct from the return of one trajectory.
- **State-value function / state value**, $V^\pi(s)$: Expected return starting in state $s$ and following policy $\pi$. Terminal state value is zero.
- **Action-value function / action value**, $Q^\pi(s,a)$: Expected return taking action $a$ in state $s$ once, then following $\pi$.
- **Prediction / policy evaluation:** Computing or estimating values for a specified policy.
- **Bellman expectation equation:** A policy's value equals expected immediate reward plus discounted next-state value, averaging environmental outcomes and actions under that policy.
- **Bellman backup:** An update combining immediate rewards and discounted continuation estimates, with the appropriate averages or maximum.
- **Monte Carlo (MC):** Estimation using sampled outcomes; here, averaging sampled returns to estimate policy values.
- **Suffix return:** The return starting at a particular visit within a recorded trajectory; discount exponents restart at that visit.
- **Every-visit Monte Carlo:** Each visit to a state contributes a suffix return, including repeated visits within one episode. Those returns can be correlated.
- **Iterative policy evaluation:** Repeated Bellman expectation backups using a known model until a stopping condition is met.
- **Sweep:** One update for every state in the value table.
- **Synchronous sweep:** All new values read from the previous value table.
- **In-place update:** An update that can use values already changed during the same sweep.
- **Tolerance**, $\theta$: A threshold on the maximum change in a sweep used to stop iteration; it is not directly the value error.
- **Bellman residual:** The largest absolute disagreement between a value table and its Bellman backup under the specified operator.
- **Value error:** The difference between an estimated value table and the true value table; for a discounted contraction, residual $b$ gives a maximum-error bound $b/(1-\gamma)$.
- **Sampling error:** Variation in an estimate caused by finite random samples.
- **Truncation bias:** Systematic error from handling unobserved continuation as zero or selecting only completed trajectories. More samples alone do not remove it.
- **Bootstrap:** Use an estimated continuation value rather than waiting for the complete future return; introduced as motivation for later temporal-difference (TD) learning.

## Bandits and exploration

Source: [Lesson 3](../lessons/RL-lesson-03-bandits-exploration-01.md) and its
[hands-on tutorial](../lessons-hands-on/RL-lesson-03-bandits-exploration-02-hands-on.md).

- **Multi-armed bandit:** Repeated choice among options with unknown reward distributions, without next-state transitions to track.
- **Arm / pull:** An arm is an available option; a pull is one selection and reward observation.
- **True action value**, $q_*(a)$: An arm's expected immediate reward. This bandit quantity differs from the MDP action value $Q^\pi(s,a)$.
- **Action-value estimate**, $Q_t(a)$: The learner's estimate of an arm's true mean before pull $t$.
- **Stationary bandit:** Each arm's reward distribution remains fixed over time.
- **Independent draws conditional on the chosen arm:** Previous rewards do not change the distribution of the next reward for that arm.
- **Exploration:** Trying options to gather information that may improve later decisions.
- **Exploitation:** Selecting an arm with the largest current estimate.
- **Greedy action / greedy rule:** An action attaining the largest current action-value estimate, or a rule selecting such an action.
- **Maximum / argmax:** `max` gives the largest score; $\arg\max$ gives the action or actions attaining it.
- **Sample average:** The mean of observed rewards from one arm, using that arm's own count.
- **Incremental update:** Updating an estimate as old estimate plus step size times prediction error, $Q\leftarrow Q+\alpha(R-Q)$.
- **Prediction error**, $R-Q$: Observed reward minus the old estimate in the bandit update.
- **Step size**, $\alpha$: The fraction of prediction error used in an update. $\alpha_n=1/n$ yields the sample average; fixed $\alpha$ weights recent observations more strongly. It is distinct from a decision step.
- **Epsilon-greedy**, $\epsilon$: Explore uniformly among all arms with probability $\epsilon$; otherwise exploit, breaking ties uniformly. Exploration can select a greedy arm too.
- **Tie-breaking:** Choosing among equally scoring actions. Lesson 3 uses random ties; Lesson 4 chooses left on exact ties.
- **Optimistic initialization:** Starting estimates high to encourage visiting options whose estimates remain high; it does not guarantee adequate exploration.
- **Regret:** Performance shortfall relative to a benchmark, usually a true best arm.
- **Cumulative pseudo-regret:** Sum of true-mean gaps, $\sum_{t=1}^T[q_*(a_*)-q_*(A_t)]$. It is nonnegative and excludes realized reward noise.
- **Realized reward shortfall:** $Tq_*(a_*)-\sum_tR_t$; it can be negative after lucky rewards and differs from pseudo-regret in a single run.
- **Optimal action / true best arm**, $a_*$: An arm attaining the largest true mean; ties can give several optimal actions.
- **Fraction of optimal actions:** Proportion of pulls selecting a true best arm.
- **Run:** One independent learner experiment containing a sequence of pulls; it is distinct from one pull or a LineWorld episode.
- **Mean reward:** Average observed reward per pull, averaged across runs when reporting a strategy.
- **Standard deviation:** Spread of observations, here the run-average rewards.
- **Standard error (SE):** Uncertainty in the estimated mean across independent runs; estimated as their sample standard deviation divided by the square root of the number of runs.
- **Gaussian reward noise:** Normally distributed variation around an arm's true mean; the supplied lab uses standard deviation one.
- **Horizon:** The number of remaining decisions, which affects the benefit of gathering information.

Bandit notation uses $R_t$ for the reward on pull $t$, unlike $R_{t+1}$ for the
reward following MDP action $A_t$. Arm-specific update count $n$ differs from the
total pull count $t$.

## Dynamic programming control

Source: [Lesson 4](../lessons/RL-lesson-04-dynamic-programming-control-01.md) and its
[hands-on tutorial](../lessons-hands-on/RL-lesson-04-dynamic-programming-control-02-hands-on.md).

- **Control:** Finding or improving a policy, rather than evaluating a fixed one.
- **Dynamic programming (DP):** Planning with a known model by repeatedly combining immediate rewards and continuation values.
- **One-step lookahead**, $Q_V(s,a)$: The expected immediate reward plus discounted continuation using the supplied table $V$. It equals $Q^\pi$ when $V=V^\pi$.
- **Policy improvement:** Choosing actions greedy with respect to the old policy's action values. Exact evaluation guarantees nondecreasing value, rather than strict improvement everywhere.
- **Policy iteration:** Alternate policy evaluation and greedy improvement until the policy stabilizes.
- **Improvement round:** One evaluation-and-improvement cycle in policy iteration; its evaluation can require many sweeps.
- **Optimal value**, $V_*(s)$: The greatest expected return achievable from state $s$.
- **Optimal policy:** A policy attaining optimal values.
- **Bellman optimality equation:** A state's optimal value is the maximum, over available actions, of expected reward plus discounted optimal continuation.
- **Value iteration:** Repeated Bellman optimality backups, followed by extracting a greedy policy.
- **Bellman operator**, $T^\pi$ or $T_*$: The mapping from a value table to its expectation or optimality backup table.
- **Maximum norm**, $\|V-W\|_\infty$: Largest absolute difference between corresponding entries of two value tables.
- **Contraction:** A mapping that shrinks maximum differences by a factor less than one; discounted Bellman operators shrink them by at most $\gamma<1$.
- **Fixed point:** A table unchanged by the operator; the policy value or optimal value is the corresponding Bellman fixed point.
- **Bellman optimality residual:** $\|V-T_*V\|_\infty$, the largest disagreement with the optimality backup. Its value-error guarantee assumes $\gamma<1$.
- **Action gap:** Difference between competing action scores, used to judge whether approximate values identify a reliable greedy action.
- **Policy quality / policy loss:** How far a selected policy's value falls below the optimal value; distinct from the error in its input value estimates.

## Mathematical tools used in the progressive exercises

Sources: the “Tools for the stretch exercises” sections of
[Lesson 1](../lessons/RL-lesson-01-rl-foundations-01.md),
[Lesson 2](../lessons/RL-lesson-02-value-functions-bellman-01.md),
[Lesson 3](../lessons/RL-lesson-03-bandits-exploration-01.md), and
[Lesson 4](../lessons/RL-lesson-04-dynamic-programming-control-01.md).

- **Expectation**, $\mathbb E$: A probability-weighted average. The vertical bar in $\mathbb E[X\mid Y]$ or $P(X\mid Y)$ means “given.”
- **Summation**, $\sum$: Add the indexed terms; the limits identify which terms to include.
- **Estimate notation**, $\widehat V$: A hat marks an estimated quantity. A superscript $\pi$ labels the policy, and $G^{(i)}$ labels sample $i$; neither is an exponent in these uses.
- **Absolute value**, $|x|$: Magnitude, ignoring sign.
- **Bound:** A worst-case limit, rather than a prediction of the exact result.
- **Bounded rewards:** Rewards whose absolute magnitude is at most a finite constant $M$.
- **Triangle inequality:** The magnitude of a sum is at most the sum of its terms' magnitudes.
- **Geometric series:** A sum of successive powers; for $0\leq\gamma<1$, $\sum_{j=0}^{\infty}\gamma^j=1/(1-\gamma)$.
- **Modulo:** The remainder after division, used to represent the phase of a repeating cycle.
- **Self-loop:** A transition returning to the same state.
- **Consistency equation:** An equation requiring mutually dependent values to agree, including when a value appears on both sides.
- **Variance**, $\sigma^2$: Spread measured by squared distances from the mean; its square root is the standard deviation. Independent fixed-count sample averages have variance $\sigma^2/n$.
- **Componentwise:** At every state separately.
- **Monotone operator:** Increasing continuation values cannot decrease the resulting probability-weighted Bellman backup.
- **Margin:** The winning action's score minus a competing action's score.

## Vocabulary in code and command output

Keep established CLI field names stable and read them with these meanings:

- `state`, `next_state`, `action`, `reward`, `terminated`: the components and ending status of a transition. `LineWorld.step()` returns next state, reward, and termination status; the runner owns truncation.
- `Episode.states`, `.actions`, `.rewards`: a recorded trajectory; states include the initial state, so there is one more state than actions or rewards.
- `Episode.discounted_return(gamma)`: recorded return, including partial return when truncated.
- Lesson 1 `reward_sum`, `return`, `steps`: means of recorded reward sums, returns, and transition counts across all recorded episodes. `success_rate` is the fraction reaching the goal; `truncated` counts truncated episodes.
- `--gamma`: discount factor; `--max-steps`: external recording limit. In Lesson 3, `--steps` counts bandit pulls.
- Lesson 2 `dp_value`, `mc_value`: computed and sampled state-value estimates for the supplied policy. `visits` counts sampled state visits. `DP iterations` counts synchronous evaluation sweeps.
- Lesson 3 `means`, `estimates`, `counts`: true arm means, learned action-value estimates, and arm-specific pull counts. `mean_pseudo_regret` averages cumulative pseudo-regret over runs; `reward_SE` is the standard error of run-average reward; `optimal_fraction` is the fraction of true optimal actions.
- Lesson 4 `action_value(...)`: one-step lookahead $Q_V$, not an independently learned $Q$ table. `rounds` counts policy improvement cycles for policy iteration, and sweeps for value iteration. `residual` is the Bellman optimality residual.
- `--seed`: initializes pseudorandom streams for repeatable experiments in the same Python environment. It is an experiment setting, not a learning rule.
