# Game Theory in Darts

## Overview

This project applies game theory, probability and numerical optimisation to the game of darts.

The analysis considers two distinct phases of play:

- **Scoring phase:** determining the optimal location to aim based on a player's throwing accuracy.
- **Endgame:** analysing the strategic decision between an aggressive checkout attempt and a safer setup shot based on the opponent's threat level.

A player's skill is modelled using the standard deviation, σ, of their throwing error. The expected score of different aiming positions is then evaluated computationally using a discretised dartboard and Gaussian error model.

## Methods

The project uses:

- Bivariate Normal distributions to model throwing accuracy
- Expected payoff and optimisation
- Numerical integration
- Discrete convolution
- Zero-sum game theory
- Backward induction
- Python, NumPy, SciPy and Matplotlib

## Results

The scoring-phase analysis shows that the optimal aiming strategy depends on the player's accuracy. For a low-variance player (σ = 5 mm), the optimal strategy is centred on the Triple 20, while for a higher-variance player (σ = 40 mm), the optimal strategy shifts towards the Triple 19 / bottom-left region of the board.

For the 50 checkout scenario considered in the endgame analysis, the model gives a critical opponent threat level of approximately 39.6%. Above this level, the aggressive Bullseye strategy becomes preferable within the assumptions of the model.

## Report

The full mathematical analysis and discussion can be found in:

[Game Theory in Darts Report](game-theory-darts-report.pdf)

## Code

The Python implementation simulates dart throws, models the dartboard geometry, calculates hit probabilities and generates expected-payoff heatmaps and strategic analysis plots.

## Limitations

The model makes several simplifying assumptions, including circular throwing variance, independent throws and a constant skill level. It therefore provides a mathematical model of dart strategy rather than a complete representation of real-world play.
