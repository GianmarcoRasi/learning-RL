# Lesson 1 — Reinforcement Learning Foundations: Agents, Environments, and MDPs — Version 2

> **Version 2 — Expanded explanations and worked practical examples.**  
> Read each concept in words, then follow its numbers through the equation. The progressive theoretical exercises are retained here; the command-line labs and practical exercises are in the companion hands-on file.  
> **Study pace:** Allow 75–100 minutes for explanations and worked examples. The progressive theory sequence can be a separate 60–90 minute session.

> **Course:** Reinforcement Learning — Theory + Practice  
> **Lesson goal:** Build the mental model that everything else in RL will use.

**Hands-on companion:** [Python labs and practical exercises](RL-lesson-01-rl-foundations-02-hands-on.md)

### How to study this lesson

On a first pass, read the explanations and redo each worked example on paper. Then try the short in-text questions. Use the companion hands-on file for the command-line lab. Work through the progressive theory sequence in order; its final levels deliberately go beyond the first-pass material. An example shows the reasoning; an exercise asks you to reproduce or extend it.

## Learning objectives

By the end of this lesson, you should be able to:

-   Explain how reinforcement learning differs from supervised learning.
-   Identify the agent, environment, state, action, reward, and trajectory in an RL problem.
-   Describe a Markov Decision Process (MDP).
-   Distinguish immediate reward from return.
-   Explain the role of the discount factor $\gamma$.

------------------------------------------------------------------------

## 1. What is reinforcement learning?

Reinforcement learning (RL) is about learning **what to do through experience when choices have consequences over time**. The **agent** is the decision maker: it chooses actions. The **environment** is the system the agent interacts with: it responds to those actions and provides feedback. Through this interaction, the agent gains experience and learns to improve its choices.

**Example**: Think of a robot vacuum. Cleaning earns a benefit, but uses battery. Charging takes time and earns no cleaning benefit immediately. A robot that always cleans whenever it can may run out of power far from its charger. A better strategy sometimes accepts a small cost now to keep working later.

This is **sequential decision making**: one decision changes the situation in which the next decision happens. RL often involves rewards that arrive only after several actions, so the learner must work out which earlier decisions helped.

### How is this different from supervised learning?

In supervised learning, a training example usually includes a target answer, such as “this picture shows a cat.” In RL, feedback usually says how an outcome scored, without giving the best action as a label. For **example**, A failed cleaning run does not directly tell the robot when it should have charged.

An RL agent also affects what it gets to observe, for **example**, visiting the charger teaches it different things from cleaning another room. In the companion hands-on lab, we only simulate this interaction. Learning a policy comes later.

## 2. The interaction loop: a decision log

At each decision step, the agent receives information describing the current **state** and chooses an **action** from the available choices. In response, the environment produces a **next state** and a **reward**.

A state describes the current situation using the variables included in our model. An action is a choice the agent can make. A reward is a numerical signal that scores the resulting transition; it does not directly tell the agent which action was best.

The index $t$ counts decision steps. We write the current state as $S_t$ and the chosen action as $A_t$. After that action, the environment produces reward $R_{t+1}$ and next state $S_{t+1}$:

$$
S_t\rightarrow A_t\rightarrow (R_{t+1},S_{t+1}).
$$

The reward is numbered $t+1$ because it results from action $A_t$. At decision zero, the agent observes $S_0$, chooses $A_0$, and then receives $R_1$ and $S_1$.

For **example**, consider one robot decision:

| Part | Concrete meaning | Symbol |
| --- | --- | --- |
| Current state | In the kitchen, low battery | $S_t$ |
| Action | Return to charger | $A_t$ |
| Reward | Score $-1$ for time spent charging | $R_{t+1}$ |
| Next state | At charger, high battery | $S_{t+1}$ |

A **transition** is one interaction step: the agent chooses an action in the current state, and the environment produces a reward and the next state. We record it as $(S_t,A_t,R_{t+1},S_{t+1})$.

A **trajectory**, written $\tau$, is a sequence of connected transitions: each transition’s next state becomes the following transition’s current state. It can describe a finite segment, a complete episode, or an ongoing sequence:

$$
\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\ldots).
$$

For **example**, “low battery, charge, $-1$, high battery, clean, $+3$, low battery” is a two-transition trajectory.

### Episodes and continuing tasks

An **episode** is an interaction sequence with a task-defined end, such as reaching an exit or failing. A **continuing task** has no natural end, such as, for **example**, managing heating indefinitely. In an **episodic task**, entering a **terminal state** ends the episode: no actions or rewards follow. Any success or failure reward belongs to the transition entering that state.

### A policy is a decision rule

A **policy**, written $\pi$, tells the agent how to choose in each state. For **example**, “Charge whenever battery is low” is a deterministic policy: it specifies one action. A stochastic policy gives probabilities instead:

$$
\pi(a\mid s)=P(A_t=a\mid S_t=s).
$$

Read the vertical bar as “given.” The formula says: **the chance of selecting action $a$, given state $s$**.

For **example**, If $\pi(\text{charge}\mid\text{low})=0.8$ and $\pi(\text{clean}\mid\text{low})=0.2$, the robot charges in about 80 out of 100 such decisions, over many repetitions. Those counts are averages, not a guarantee for a particular group of 100. The available action probabilities must add to one.

The policy supplies the action choices along a trajectory. The same policy can produce different trajectories when its choices or the environment’s outcomes are random.

**Example**: Suppose the robot follows the deterministic policy “always clean.” Starting with low battery, one run might produce “low battery, clean, $+3$, low battery,” while another produces “low battery, clean, $-10$, failed.” The policy chooses the same action in both runs, but the environment’s random response produces different trajectories.

## 3. Markov Decision Processes: specifying the world

A **Markov Decision Process (MDP)** is a mathematical model of the agent–environment interaction. It specifies the possible states, the actions available in each state, the probabilities of transitions, and how rewards are generated. Its defining requirement is the **Markov property**: the current state and chosen action contain enough information to determine the probabilities of the next state and reward.

We often list its ingredients as

$$
(\mathcal S,\mathcal A,P,R,\gamma).
$$

| Ingredient | Meaning |
| --- | --- |
| $\mathcal S$ | Set of possible states |
| $\mathcal A$ | Set of actions; available actions may depend on the state |
| $P(s'\mid s,a)$ | Probability of next state $s'$ after action $a$ in state $s$ |
| $R$ | Rule or distribution that generates rewards |
| $\gamma$ | Discount factor controlling the weight of future rewards |

### Worked example: specifying the robot vacuum MDP

For the robot vacuum, use three states: high battery ($H$), low battery ($L$), and failed ($F$). The available actions are clean and charge in either battery state; no actions are available after failure. Use $\gamma=0.9$. The following table specifies the transition probabilities and rewards:

| State | Action | Next state | Probability | Reward |
| --- | --- | --- | ---: | ---: |
| $H$ | Clean | $L$ | $1$ | $3$ |
| $H$ | Charge | $H$ | $1$ | $-1$ |
| $L$ | Charge | $H$ | $1$ | $-1$ |
| $L$ | Clean | $L$ | $0.5$ | $3$ |
| $L$ | Clean | $F$ | $0.5$ | $-10$ |

For each state-action pair, outcome probabilities sum to one. The two low-battery cleaning rows describe possible outcomes of **one** choice, not two choices. $F$ is terminal: failure ends this task, with no later actions or rewards.

Rewards may depend on the next state as well as the current state and action. In the robot example, cleaning at low battery earns $+3$ if the robot keeps working, but $-10$ if it fails. The two rows specify both the possible next states and their associated rewards.

### The Markov property: does the state tell enough?

A Markov state contains the information needed to predict the distribution of the next outcome once the action is known. Older history adds nothing further to that prediction:

$$
P(S_{t+1}\mid S_t,A_t,S_{t-1},A_{t-1},\ldots)
=P(S_{t+1}\mid S_t,A_t).
$$

The left side uses both present and history. The right side uses only present state and action. Equality says that a well-chosen state already summarizes the relevant history. In a full MDP, this principle applies to the reward and next-state outcome together.

Markov does **not** mean predictable with certainty. The next outcome can be random; the requirement is that its probabilities depend only on the current state and action, without needing additional history. If the agent cannot observe an important variable, its observation may not provide a full Markov state.

**Example**: two robots are both in the kitchen. One has a full battery and the other has an almost empty battery. If state means only “kitchen,” the same cleaning action can have very different failure chances depending on earlier battery use. Include battery level in the state so this information is available now. Include other variables too if they affect the model, such as whether the floor is still dirty.

**Example**: A game enemy attacking every third turn gives another example. Equal HP does not tell us whether the next attack is due. Adding the completed-turn count modulo three distinguishes the three attack phases.

### Task deadlines and simulation cutoffs

Stopping an experiment after 100 steps is **truncation** if the actual task would continue. It stops our record, not the underlying consequences. A genuine task deadline is different: time remaining must be included in the state if it changes the best decision.

## 4. Rewards and return: scoring an entire plan

A reward scores one transition. A **return**, $G_t$, combines the rewards from decision $t$ onward. It is what we use to compare entire futures.

For an episode ending at decision time $T$,

$$
G_t=\sum_{k=0}^{T-t-1}\gamma^kR_{t+k+1}
=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots.
$$

The summation symbol $\sum$ means “add the terms.” $k$ starts at zero for the next reward, then increases for each reward further in the future. $\gamma^k$ is its weight. We give the first reward weight one, including when $\gamma=0$.

### What does the discount factor do?

The **discount factor**, $\gamma\in[0,1]$, controls how much weight future rewards receive. When $\gamma=0$, only the next reward counts. For $0<\gamma<1$, rewards further in the future receive progressively less weight. When $\gamma=1$, all rewards receive equal weight. We choose $\gamma$ as part of the objective.

For a continuing task the sum may be infinite:

$$
G_t=\sum_{k=0}^{\infty}\gamma^kR_{t+k+1}.
$$

If rewards are bounded and $\gamma<1$, decreasing weights keep this sum finite. Earning $1$ forever with $\gamma=0.9$ gives $1+0.9+0.81+\cdots=1/(1-0.9)=10$. With $\gamma=1$ that same sum grows without bound. Finite episodes can use $\gamma=1$; infinite tasks need more care.

### Expected return

When outcomes are random, compare **expected return**: a probability-weighted average over possible futures. A one-step action paying $10$ with probability $0.8$ and $-10$ with probability $0.2$ has expected return $0.8(10)+0.2(-10)=6$. It does not pay $6$ on any single attempt, and this average alone does not describe its risk.

### Worked example: cleaning or charging

Suppose the robot starts with low battery. We compare the rewards over the next two transitions, using $\gamma=1$, so both rewards receive equal weight.

First, consider two possible trajectories:

| Actions and outcomes | First reward | Second reward | Return $G_0$ |
| --- | ---: | ---: | ---: |
| Clean successfully, then clean and fail | $+3$ | $-10$ | $-7$ |
| Charge, then clean successfully | $-1$ | $+3$ | $+2$ |

Charging gives a worse immediate reward, but a higher return than the failed cleaning trajectory. The reward scores one transition; the return combines rewards across the future. These numbers are a designer's scoring rule; they are not automatically the robot's energy measurements.

**This comparison describes two trajectories.** Deciding which plan is better requires comparing expected returns across all its possible outcomes, including failure on the first cleaning action.

### Worked example: delivering a parcel

A courier makes two ordinary moves and then completes a delivery. The rewards are $-1,-1,+10$, and $\gamma=0.9$.

| Reward | Delay from the first reward | Weight | Contribution to $G_0$ |
| --- | ---: | ---: | ---: |
| $R_1=-1$ | $0$ | $1$ | $-1$ |
| $R_2=-1$ | $1$ | $0.9$ | $-0.9$ |
| $R_3=10$ | $2$ | $0.9^2=0.81$ | $8.1$ |

Thus

$$
G_0=-1+0.9(-1)+0.81(10)=6.2.
$$

The ordinary reward sum is $8$, but the discounted return is $6.2$. From the next decision, $G_1=-1+0.9(10)=8$; from the final decision, $G_2=10$. Discount exponents restart relative to the point from which we are evaluating the future.

Using the same delivery rewards, changing the discount factor changes the return:

| Discount | Delivery return | Interpretation |
| --- | ---: | --- |
| $\gamma=0$ | $-1$ | Only the next reward counts |
| $\gamma=0.5$ | $-1-0.5+2.5=1$ | Distant rewards lose weight quickly |
| $\gamma=0.9$ | $6.2$ | Distant rewards keep more weight |
| $\gamma=1$ | $8$ | Add all rewards without discounting |

### Reward design

Finally, the agent optimizes the reward rule we supply. Paying a courier $+1$ for every extra move can encourage wandering; charging a movement cost and paying for delivery can favor finishing sooner. Always ask whether a high return actually describes the behavior you want.

## 5. Worked example: LineWorld — an MDP from start to finish

LineWorld connects the state, action, transition, policy, and return concepts in one small example. Imagine a courier walking along five numbered stops: position $0$ is the depot, positions $1,2,3$ are intermediate stops, and position $4$ is the destination.

### Specify the MDP

| Ingredient | LineWorld definition |
| --- | --- |
| States | Positions $\mathcal S=\{0,1,2,3,4\}$; start at $0$ |
| Actions | Left or right at each nonterminal position |
| Left transition | From position $s<4$, move to $\max(0,s-1)$ |
| Right transition | From position $s<4$, move to $s+1$ |
| Reward | $-1$ unless the move enters $4$, in which case $+10$ |
| Terminal condition | Entry to position $4$ ends the episode; no actions or rewards follow |
| Discount | Use $\gamma=0.9$ for this example |

Every transition is deterministic: the specified next position has probability one. Moving left at $0$ keeps the courier at $0$, but still counts as a transition and earns $-1$. The goal-entry reward is $10$ total, rather than $10-1$.

**Why is position a Markov state?** Given the current position and action, these rules determine both the next position and reward. The route taken to reach the current position changes neither outcome. Position is sufficient under this model because there is no battery, attack clock, or task deadline affecting the rules.

### Follow a policy and compute its return

Choose the deterministic policy “always move right.” Its trajectory is

$$
\tau=(0,\text{right},-1,1,\text{right},-1,2,\text{right},-1,3,\text{right},10,4).
$$

There are four transitions. The final state is terminal, and the four rewards are $-1,-1,-1,10$. Their ordinary sum is $7$, while their discounted return is

$$
G_0=-1+0.9(-1)+0.9^2(-1)+0.9^3(10)
=-1-0.9-0.81+7.29=4.58.
$$

The policy supplies the choices; the MDP supplies their consequences; the return scores the resulting future. This policy is specified in advance—nothing is learning yet.

Implement this environment and compare fixed and random policies in the [hands-on companion](RL-lesson-01-rl-foundations-02-hands-on.md). That lab also explores simulation cutoffs and alternative reward rules.

------------------------------------------------------------------------

## 6. Theory exercise --- Compute returns

Suppose the future rewards are

$$
R_1=2,\quad R_2=-1,\quad R_3=5
$$

and the episode ends after $R_3$.

### Exercise 1A

Calculate $G_0$ for:

1.  $\gamma=0$
2.  $\gamma=0.5$
3.  $\gamma=1$

Do this before reading the solution.

**Solution**

Using

$$
G_0=R_1+\gamma R_2+\gamma^2R_3,
$$

for $\gamma=0$:

$$
G_0=2.
$$

For $\gamma=0.5$:

$$
G_0=2+0.5(-1)+0.5^2(5)=2.75.
$$

For $\gamma=1$:

$$
G_0=2-1+5=6.
$$


### Exercise 1B --- Reasoning

Two actions are available:

-   Action A: reward `+4` immediately, then the episode ends.
-   Action B: reward `0` now and `+10` one step later.

For what values of $\gamma$ is B preferable to A?

**Solution**

Action A has return

$$
G_A=4.
$$

Action B has return

$$
G_B=10\gamma.
$$

B is preferable when

$$
10\gamma > 4,
$$

therefore

$$
\gamma>0.4.
$$

At $\gamma=0.4$, the two actions have equal return.



------------------------------------------------------------------------

## 7. Theory exercise --- Model an RL problem

Choose one problem:

-   robot vacuum;
-   game combat AI;
-   stock inventory control;
-   maze navigation;
-   traffic-light control.

Define:

$$
\mathcal{S},\quad
\mathcal{A},\quad
R,\quad
\text{terminal conditions}.
$$

Then answer:

1.  Is your proposed state Markov?
2.  What important information might be missing?
3.  Could your reward function create unintended behavior?
4.  Is the task episodic or continuing?

------------------------------------------------------------------------

## 8. Progressive theoretical exercises — Foundations

Work through these exercises **in order**, after reading the lesson. The earlier in-text exercises are warm-ups; this sequence gradually removes the scaffolding. No programming is needed. Write your reasoning as well as your final answer. Allow an additional 60–90 minutes, or split the sequence across study sessions. The final two exercises are stretch challenges: use their hints when needed. Try each question before reading its solution.

### Tools for the stretch exercises

A bound is a worst-case limit, not a prediction of the exact result. $|x|$ means magnitude: $|-3|=3$. If rewards have magnitude at most $M$ and $\gamma<1$, their weighted magnitudes fit under the geometric sum $M/(1-\gamma)$. For example, $M=2$ and $\gamma=0.5$ give a total-magnitude bound of $4$. The triangle inequality says the magnitude of a sum is at most the sum of its magnitudes.

“Modulo three” means the remainder after division by three: counts $0,3,6$ have phase zero. It is a compact way to remember a repeating three-turn cycle.

### Level 1 — Guided application

### 1T1 — Read a trajectory

A trajectory is $(s_0,\text{right},-1,s_1,\text{right},10,\text{terminal})$.

1. Identify $S_0,A_0,R_1,S_1,A_1,R_2$.
2. For $\gamma=0.5$, calculate $G_0$ and $G_1$.
3. Does the final reward belong to the transition into the terminal state or to a later action there?

**Hint**

Write the two rewards in their time order. The first reward has discount exponent zero.



**Solution**

$S_0=s_0$, $A_0=\text{right}$, $R_1=-1$, $S_1=s_1$, $A_1=\text{right}$, $R_2=10$. $G_0=-1+0.5(10)=4$ and $G_1=10$. The reward $10$ is received on entry to the terminal state; no later action is required.



### 1T2 — A stochastic policy

At a single decision state, action A terminates with reward $6$ and action B terminates with reward $-2$. A policy chooses A with probability $p$.

1. Express the expected return as a function of $p$.
2. Evaluate it at $p=0.25$ and $p=0.75$.
3. Can changing $\gamma$ change these answers? Explain.

**Hint**

Use a probability-weighted average; there is only one reward.



**Solution**

$\mathbb{E}[G_0]=6p-2(1-p)=8p-2$. The two expectations are $0$ and $4$. Discounting has no effect because both outcomes terminate after the first reward, which is undiscounted.



### Level 2 — Multi-step reasoning

### 1T3 — Repair a non-Markov state

Both characters start with $20$ HP, and the number of completed player actions starts at zero. The player can `wait`, which changes neither character's HP directly. After each player action, the completed-action count increases by one; the enemy then deals $5$ damage to the player if the new count is divisible by three, and $0$ otherwise. The enemy's HP stays unchanged. Assume there are no other variables affecting the dynamics.

The proposed state is $(\text{player HP},\text{enemy HP})$. Compare two histories: before any player actions, and after two consecutive waits. In both cases, the proposed state is $(20,20)$.

1. If the next action is `wait` in both situations, what damage follows in each? Why does this show that the proposed state is not Markov?
2. Add a clock variable with three possible values that distinguishes the phases. Define its meaning and update rule, and explain why all three phases matter.
3. Suppose instead that damage is drawn independently each turn from the same fixed distribution, with no attack clock. Does randomness alone make the HP state non-Markov?

**Hint**

Write the new completed-action count after the next wait in each history. For the clock variable, remember only the remainder after dividing the current completed-action count by three.



**Solution**

The histories give different outcomes for the **same current HP state and the same next action**:

| History | Current completed-action count | Count after the next wait | Enemy damage | Next HP state |
| --- | ---: | ---: | ---: | --- |
| No actions yet | $0$ | $1$ | $0$ | $(20,20)$ |
| Two waits completed | $2$ | $3$ | $5$ | $(15,20)$ |

Knowing only $(20,20)$ and `wait` does not determine the next outcome; knowing the history changes the prediction. The proposed state therefore omits relevant information.

Let $c=n\bmod3$, where $n$ is the number of player actions already completed. The augmented state is $(\text{player HP},\text{enemy HP},c)$, with $c\in\{0,1,2\}$. On the next action, damage is $5$ if the **current** phase is $c=2$ and $0$ otherwise; update the phase to $c'=(c+1)\bmod3$.

Phases $0$ and $1$ both give zero damage on the next wait, but they are not interchangeable: from phase $0$ the next attack is three actions away, while from phase $1$ it is two actions away. A three-value phase records the cycle without retaining the full count. Under the stated assumptions, HP and this phase determine the next-outcome distribution, so the augmented state is Markov.

Independent random damage from a fixed distribution is compatible with a Markov HP state under the same assumptions. The next outcome can be uncertain while its probabilities depend only on the current state and action. Dependence on forgotten history, rather than randomness itself, is the problem.



### 1T4 — Discounting changes a decision

Action A gives $3$ immediately and ends the episode. Action B gives $-1$ immediately and $8$ on the next transition, then terminates. Action C gives $0,0,12$ on three successive transitions, then terminates. No later choices occur.

Find the preferred action for **every** $\gamma\in[0,1]$, including ties. Show how you exclude an action that never wins.

**Hint**

Compare $3$, $-1+8\gamma$, and $12\gamma^2$ in pairs.



**Solution**

A and B tie at $\gamma=1/2$. A and C also tie at $\gamma=1/2$. For $\gamma<1/2$, B and C both lie below A. For $\gamma>1/2$, C exceeds A and
$$12\gamma^2-(-1+8\gamma)=12\gamma^2-8\gamma+1=(6\gamma-1)(2\gamma-1)>0.$$
Thus A wins for $0\leq\gamma<1/2$, all three tie at $1/2$, and C wins for $1/2<\gamma\leq1$. B never wins uniquely, although it participates in the three-way tie.



### Level 3 — Derivation and critique

### 1T5 — Bound the infinite return

Suppose every reward satisfies $|R_t|\leq M$, with $M>0$ and $0\leq\gamma<1$.

1. Derive a bound on $|G_0|$.
2. If you keep only the first $H\geq1$ rewards, bound the magnitude of the omitted tail.
3. Evaluate the tail bound for $M=2$, $\gamma=0.5$, $H=5$.
4. Why does the argument fail at $\gamma=1$ for a continuing task?

**Hint**

Use the geometric-series identity $\sum_{j=0}^{\infty}\gamma^j=1/(1-\gamma)$ and the triangle inequality.



**Solution**

The triangle inequality gives
$$|G_0|\leq\sum_{j=0}^{\infty}\gamma^j M=\frac{M}{1-\gamma}.$$
The omitted rewards begin at exponent $H$, hence their magnitude is at most
$$\frac{M\gamma^H}{1-\gamma}.$$
The numerical bound is $2(0.5)^5/(0.5)=0.125$. At $\gamma=1$ the geometric series does not converge. Bounded individual rewards alone do not bound their infinite sum; a continuing reward of $2$ per step diverges.



### 1T6 — Reward design can reverse an objective

Set $\gamma=1$. A successful short route takes two transitions; a successful long route takes five. Only the final transition earns the goal reward $10$; each preceding transition earns a movement bonus $b$. There are no other rewards.

1. Derive both route returns.
2. Which route is preferred when $b=1$, $b=0$, and $b=-1$?
3. If the agent can loop for $n$ additional nonterminal transitions before completing the short route, what happens as $n$ grows when $b>0$?
4. Propose a reward definition that favors earlier successful completion.

**Hint**

Count nonterminal transitions separately from the goal-entry transition.



**Solution**

The short return is $b+10$; the long return is $4b+10$. At $b=1$, the long route wins ($14$ versus $11$). At $b=0$, both give $10$. At $b=-1$, the short route wins ($9$ versus $6$). After $n$ extra transitions the short-route return is $(n+1)b+10$, unbounded as $n\to\infty$ if $b>0$. A movement bonus can therefore reward delay indefinitely. A negative reward on each nonterminal transition with a positive goal-entry reward favors shorter successful paths. Alternatively, sparse positive goal reward with $0<\gamma<1$ favors earlier completion among otherwise identical successful trajectories.



### Level 4 — Synthesis challenges

### 1T7 — Design a complete small MDP

A robot has two battery states, high ($H$) and low ($L$), plus a terminal failed state ($F$). Use $\gamma=0.9$.

At $H$, `work` earns $2$ and moves to $L$; `charge` earns $-1$ and stays at $H$. At $L$, `work` earns $2$ and fails with probability $0.5$, otherwise staying at $L$; `charge` earns $-1$ and moves to $H$.

1. Write every state-action-outcome probability and reward in a table. State the terminal convention.
2. Compare expected **two-transition truncated returns** from $H$ for work-then-charge and work-then-work, with zero continuation after the second transition.
3. Explain why this comparison alone does not settle the infinite-horizon optimal policy.

**Hint**

In the two-step comparison, both possible outcomes of the second work action earn the same immediate reward.



**Solution**

| State | Action | Next state | Probability | Reward |
| --- | --- | --- | ---: | ---: |
| H | work | L | 1 | 2 |
| H | charge | H | 1 | -1 |
| L | work | F | 0.5 | 2 |
| L | work | L | 0.5 | 2 |
| L | charge | H | 1 | -1 |

At $F$, no decisions or further rewards occur. Each nonterminal state-action row group has total probability one. Work-then-charge gives $2+0.9(-1)=1.1$. Work-then-work gives $2+0.9(2)=3.8$. The latter looks better over two transitions, but failure removes future earning opportunities. Infinite-horizon comparison must account for continuation at the second transition's resulting state. A truncated reward sum does not include that consequence.



### 1T8 — A task deadline belongs in the state

A task allows at most two transitions. At physical location $x$, `cash` terminates with reward $3$, while `travel` earns $0$ and reaches $y$. At $y$, `collect` terminates with reward $8$. If the deadline is reached before collection, the task ends with no extra reward. Let $\gamma=1$.

1. What should the agent choose at $x$ with two transitions remaining? With one remaining?
2. Explain why location alone is insufficient for a stationary policy here.
3. Specify an augmented state and how its time component changes.
4. Contrast this task deadline with an external safety limit used to interrupt simulation of a continuing task.

**Hint**

The same location can require different actions at different remaining horizons.



**Solution**

With two transitions remaining, travel then collect gives $8>3$. With one remaining, travel yields $0$ and ends the task, so cash wins with $3$. Location alone cannot encode the differing optimal choices or remaining task dynamics. Use $(\text{location},h)$, where $h\in\{1,2\}$ is transitions remaining. Each transition decreases $h$ by one; reaching zero terminates the task, as does cash or collection. A deadline is part of the objective and environment. An external simulation safety limit merely stops observation; it does not make the underlying task terminal or justify assigning its remaining return zero.




## 9. Key ideas

The fundamental RL loop is:

$$
\text{state}
\rightarrow
\text{action}
\rightarrow
\text{reward + next state}.
$$

An MDP provides the mathematical framework.

The agent follows a policy $\pi$, and the objective is generally to maximize expected **return**, not immediate reward.

This distinction between **reward now** and **consequences later** is the first major idea to internalize.

------------------------------------------------------------------------

## 10. Before Lesson 2

Make sure you can explain, without notes:

-   What is a policy?
-   What is a trajectory?
-   What makes a state Markov?
-   What is the difference between reward and return?
-   What does $\gamma$ do?

In Lesson 2, we give the agent a way to evaluate states and actions using **value functions and Bellman equations**.
