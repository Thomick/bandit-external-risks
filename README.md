# Multi-armed bandits with external risks
*by Thomas Michel*

Repository containing my experiments with bandits affected by external hazards. Realized during my 2023 internship at Inria Lille in the Scool team.

## Structure
- Definitions for standard bandits and a few basic algorithms in `src/bandit`
- Definitions for bandits with external risks and adapted algorithms in `src/forecastbandit`
- Experiments for bandit with external risks in `src/experiment_forecast.py`

## Usage
No particular module is used other than very common ones.

A few experiments can be toggled on and off in `src/experiment_forecast.py`. Once the experiment is set up, just run the script with python3.
