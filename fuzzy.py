import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def calculate_reliability(confidence):

    # Input
    ai_confidence = ctrl.Antecedent(
        np.arange(0, 101, 1),
        "ai_confidence"
    )

    # Output
    reliability = ctrl.Consequent(
        np.arange(0, 101, 1),
        "reliability"
    )

    # Membership functions
    ai_confidence["low"] = fuzz.trapmf(
        ai_confidence.universe,
        [0, 0, 30, 50]
    )

    ai_confidence["medium"] = fuzz.trimf(
        ai_confidence.universe,
        [30, 55, 80]
    )

    ai_confidence["high"] = fuzz.trapmf(
        ai_confidence.universe,
        [60, 80, 100, 100]
    )

    reliability["low"] = fuzz.trapmf(
        reliability.universe,
        [0, 0, 30, 50]
    )

    reliability["medium"] = fuzz.trimf(
        reliability.universe,
        [30, 55, 80]
    )

    reliability["high"] = fuzz.trapmf(
        reliability.universe,
        [60, 80, 100, 100]
    )

    # Fuzzy rules
    rule1 = ctrl.Rule(
        ai_confidence["low"],
        reliability["low"]
    )

    rule2 = ctrl.Rule(
        ai_confidence["medium"],
        reliability["medium"]
    )

    rule3 = ctrl.Rule(
        ai_confidence["high"],
        reliability["high"]
    )

    # Control system
    system = ctrl.ControlSystem([
        rule1,
        rule2,
        rule3
    ])

    simulation = ctrl.ControlSystemSimulation(system)

    # Input confidence
    simulation.input["ai_confidence"] = confidence

    # Calculate
    simulation.compute()

    return simulation.output["reliability"]