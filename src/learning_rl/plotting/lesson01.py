"""Plots of recorded LineWorld episodes; Matplotlib is imported on demand."""

from statistics import mean


def plotting():
    # Text runs need no third-party packages.
    import matplotlib.pyplot as plt
    return plt


def plot_trajectory(episode, goal, label="episode"):
    fig, ax = plotting().subplots()
    ax.step(range(len(episode.states)), episode.states, where="post")
    ax.plot(range(len(episode.states)), episode.states, "o", markersize=4)
    ending = "terminated" if episode.terminated else "truncated"
    ax.scatter(len(episode.actions), episode.states[-1],
               marker="*" if episode.terminated else "X", s=130, label=ending, zorder=3)
    ax.axhline(goal, linestyle="--", color="gray", label="goal")
    ax.set(xlabel="Decision step t", ylabel="Position S_t",
           title=f"Trajectory: {label}", yticks=range(goal + 1))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_rewards(episode, gamma, label="episode"):
    fig, axes = plotting().subplots(2, 1, sharex=True, figsize=(7, 6))
    cumulative, total = [0.0], 0.0
    for t, reward in enumerate(episode.rewards):
        total += gamma ** t * reward
        cumulative.append(total)
    axes[0].bar(range(1, len(episode.rewards) + 1), episode.rewards)
    axes[0].set(ylabel="Reward R_n", title=f"Rewards: {label}")
    axes[1].plot(range(len(cumulative)), cumulative, "o-")
    axes[1].set(xlabel="Completed transitions n", ylabel="Cumulative discounted reward",
                title=f"From start through n transitions (gamma={gamma:g})")
    fig.tight_layout()
    return fig


def plot_comparison(experiments, gamma):
    fig, axes = plotting().subplots(1, 3, figsize=(12, 4))
    labels = list(experiments)
    for ax, values, title in (
        (axes[0], [[e.discounted_return(gamma) for e in es] for es in experiments.values()],
         "Recorded return"),
        (axes[1], [[len(e.actions) for e in es] for es in experiments.values()],
         "Recorded episode length"),
    ):
        ax.boxplot(values)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        ax.set_title(title)
    axes[2].bar(labels, [mean(e.terminated for e in es) for es in experiments.values()])
    axes[2].set(title="Success rate", ylim=(0, 1.05))
    counts = ", ".join(f"{label}: {sum(e.truncated for e in es)}/{len(es)} truncated"
                       for label, es in experiments.items())
    fig.suptitle(f"Fixed-policy comparison, gamma={gamma:g}\n{counts}")
    fig.tight_layout()
    return fig


def plot_discount(episode, label="episode"):
    fig, ax = plotting().subplots()
    gammas = [i / 100 for i in range(101)]
    ax.plot(gammas, [episode.discounted_return(g) for g in gammas])
    ax.set(xlabel="Discount gamma", ylabel="Recorded return",
           title=f"One recorded trajectory, different discounts: {label}")
    fig.tight_layout()
    return fig


def plot_running_mean(experiments, gamma):
    fig, ax = plotting().subplots()
    for label, episodes in experiments.items():
        averages, total = [], 0.0
        for n, episode in enumerate(episodes, 1):
            total += episode.discounted_return(gamma)
            averages.append(total / n)
        ax.plot(range(1, len(episodes) + 1), averages, label=label)
    ax.set(xlabel="Episodes observed", ylabel="Running mean recorded return",
           title="Performance estimates for fixed policies (no learning)")
    ax.legend()
    fig.tight_layout()
    return fig


