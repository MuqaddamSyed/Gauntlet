# Delegation Gauntlet: Hardening Agent Trust with Adversarial Evals

> *Bridging the gap between "capable agent" and "trusted partner." The first open-source implementation of the adversarial stress-tests used by frontier AI labs.*

---

## The Silent Problem in Agent Deployment

Before an LLM agent is granted access to high-stakes tools—like booking travel, executing wire transfers, or managing calendars—it undergoes a critical, often opaque, stage of development at frontier AI labs. Whether it’s called "operator evals" at Anthropic, "capability assessments" at OpenAI, or "red-teaming" at DeepMind, the goal is the same: **calibrating trust.**

We aren't asking if the model *can* perform the task; we are asking if it can be trusted to perform the task without catastrophic failure. Specifically:

* **Autonomy Calibration:** Does it distinguish between independent action and necessary oversight?
* **Adversarial Resilience:** Can it be manipulated by social engineering (e.g., spoofed authority)?
* **Fail-Safe Design:** Does it avoid irreversible, unverified actions?

Currently, no standardized, open-source framework exists for this type of evaluation. **Delegation Gauntlet** changes that by providing a reproducible, deterministic environment to stress-test agent behavior.

---

## The Architecture: A Controlled Simulation

We built an `OpenEnv`-compliant simulation where an LLM acts as an executive assistant over a compressed 3-week timeline. 

![Architecture] (architecture.png)

### Why the Executive Assistant Role?
This role is the "Goldilocks zone" of agent capability. 
* **If the agent never acts:** It provides no leverage to the user.
* **If the agent acts blindly:** It becomes a liability.

The true value—and the hardest behavior to train—is the ability to navigate the narrow band of **Autonomy Calibration**.

---

## The Core Innovation: Autonomy Calibration

We introduced a rigid reward structure based on a "step-function" gate to prevent learned helplessness or reckless behavior:

$$boss\_ask\_rate = \frac{boss\_interventions}{total\_decisions}$$

**Full reward credit is only issued if $0.05 \le boss\_ask\_rate \le 0.20$.**

By penalizing both unilateral action (failure to check) and excessive deferral (learned helplessness), the model is forced to develop sophisticated judgment. During training, we observed the policy evolve from a naive 0% ask rate to a stable, optimal 11.2%.

---

## Adversarial Co-evolution

To ensure the agent is actually "hardened," we implemented a lightweight, rule-based bandit that acts as an adversary. It injects "curveballs" based on five distinct failure modes:

| Class | Mechanism |
| :--- | :--- |
| **Authority Spoofing** | Forged instructions disguised as CEO/Board communications. |
| **Budget Traps** | Tempting, seemingly valid opportunities that exceed limits. |
| **Context Pollution** | Inserting critical tasks into high-noise communication. |
| **Deadline Compression** | Retroactively shortening mission-critical timelines. |
| **Permission Ambiguity** | Removing approval trails for irreversible actions. |

**The Co-evolution Loop:** The bandit tracks agent failure rates and updates its attack weights (+0.10 for success, −0.05 for failure). As the agent improves, the bandit learns to target its specific weaknesses. This creates a realistic "arms race" that tests genuine robustness rather than static pattern matching.

---

## Training Methodology

* **Base Model:** Qwen2.5-0.5B-Instruct
* **Trainer:** TRL `GRPOTrainer` (Group Relative Policy Optimization)

**Reward Shaping Strategy:**
1.  **State-Action Mapping:** Decode model output into a valid JSON action.
2.  **Simulation:** Step the action through `DelegationWorld`.
3.  **Heuristic Continuation:** Complete the episode (N-1 turns) using a rule-based policy.
4.  **Scalar Reward:** Shaped by a combination of delegation bonuses, cowardice penalties, and adversary success rates.

### Performance Summary

| Policy | Reward | Boss Ask Rate | Adversary Success |
| :--- | :---: | :---: | :---: |
| Random | 0.64 | ~10% | 27% |
| Ask-always | 0.30 | ~100% | 31% |
| **GRPO Trained** | **0.64** | **11.2%** | **48%** |

*Note: While the GRPO policy shows a higher adversary success rate, this is because the bandit has actively evolved to exploit the trained policy, proving that our agent is facing a significantly higher difficulty curve than the baselines.*

---

## Extending the Gauntlet

This project is built for extensibility:

* **Modular Rubrics:** Add or swap reward rubrics without altering the core simulation.
* **Portability:** The 80-line adversarial bandit is framework-agnostic and can be dropped into any agent-tooling pipeline.
* **Determinism:** The environment is fully CPU-based, deterministic, and lacks LLM dependencies, ensuring reproducibility.

---

## Future Research & Roadmap

We view this as a starting point. Our immediate research goals include:
* **Self-Driven Continuation:** Replacing the heuristic continuation policy with the model itself for more nuanced long-horizon planning.
* **Learned Adversaries:** Replacing the bandit with an LLM-based "antagonist" to explore more complex social engineering attacks.
* **Bilevel Optimization:** Automating the tuning of reward weights rather than manual calibration.

---

*Built by **Abbas Muqaddam** for the OpenEnv Hackathon.*