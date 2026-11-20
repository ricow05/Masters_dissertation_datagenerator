"""Clinical trials (parent) and their registrations (child), with 30 ordered rules.

A copy of ``algo1_bench/catalog/trials.py`` (domains, rule order, meanings), kept here so
the generator does not depend on the benchmark. "r rules" always means the first r.
"""
from __future__ import annotations

from ..rules import Rule
from ..generator.fanout import Fixed
from ..generator.spec import DatasetSpec
from ..schema import two_tables

PARENT_DOMAINS = {
    "phase": ("I", "II", "III", "IV"),
    "design": ("single-arm", "parallel", "crossover"),
    "control": ("none", "placebo", "active"),
    "population": ("adult", "paediatric", "mixed"),
    "blinding": ("open", "single", "double"),
    "randomised": ("Yes", "No"),
    "sponsor": ("industry", "academic", "government"),
    "scope": ("single-centre", "multi-centre", "multinational"),
}

CHILD_DOMAINS = {
    "age_group": ("adult", "paediatric"),
    "consent": ("self", "parental"),
    "placebo": ("Yes", "No"),
    "arm": ("experimental", "comparator", "both"),
    "pregnant": ("Yes", "No"),
    "sex": ("female", "male"),
    "status": ("ongoing", "completed", "withdrawn"),
    "outcome": ("recorded", "missing"),
}

PARENT_TABLE, PARENT_KEY = "Trial", "trial_id"
CHILD_TABLE, CHILD_KEY = "Registration", "reg_id"

# name, atoms, plain-language meaning
RULES = [
    ("P1", {"design": {"single-arm"}, "control": {"placebo", "active"}},
     "A single-arm trial has no comparator arm."),
    ("P2", {"phase": {"I"}, "population": {"paediatric"}},
     "Phase I trials are not performed on a purely paediatric population."),
    ("C1", {"age_group": {"paediatric"}, "consent": {"self"}},
     "Paediatric participants cannot give consent themselves."),
    ("C2", {"age_group": {"paediatric"}, "placebo": {"Yes"}},
     "No placebo is administered to paediatric participants."),
    ("PC1", {"control": {"placebo"}, "placebo": {"No"}},
     "A registration of a placebo-controlled trial must use a placebo."),
    ("PC2", {"population": {"paediatric"}, "age_group": {"adult"}},
     "A paediatric trial cannot enrol adult participants."),
    # -- 6 rules
    ("P3", {"design": {"single-arm"}, "randomised": {"Yes"}},
     "A single-arm trial cannot be randomised."),
    ("C3", {"sex": {"male"}, "pregnant": {"Yes"}},
     "Male participants cannot be pregnant."),
    ("PC3", {"population": {"adult"}, "age_group": {"paediatric"}},
     "An adult-only trial cannot enrol paediatric participants."),
    ("PS1", {"sponsor": {"government"}, "scope": {"multinational"}},
     "A government-sponsored trial stays within one country."),
    ("CS1", {"status": {"completed"}, "outcome": {"missing"}},
     "A completed participation has a recorded outcome."),
    ("PC4", {"control": {"none"}, "placebo": {"Yes"}},
     "A trial without a control gives no placebos."),
    # -- 12 rules
    ("P4", {"design": {"crossover"}, "control": {"none"}},
     "A crossover trial needs a comparator to cross over to."),
    ("C4", {"age_group": {"paediatric"}, "pregnant": {"Yes"}},
     "Paediatric participants are not enrolled while pregnant."),
    ("PC5", {"design": {"single-arm"}, "arm": {"comparator", "both"}},
     "Every participant of a single-arm trial is in the experimental arm."),
    ("P5", {"control": {"placebo"}, "blinding": {"open"}},
     "A placebo-controlled trial cannot be open-label."),
    ("CS2", {"status": {"ongoing"}, "outcome": {"recorded"}},
     "An ongoing participation has no final outcome yet."),
    ("PC6", {"phase": {"I"}, "pregnant": {"Yes"}},
     "Pregnant participants are excluded from phase I trials."),
    # -- 18 rules
    ("P6", {"design": {"single-arm"}, "blinding": {"double"}},
     "A single-arm trial cannot be double-blind."),
    ("C5", {"arm": {"both"}, "age_group": {"paediatric"}},
     "Paediatric participants do not go through both arms."),
    ("PC7", {"design": {"single-arm", "parallel"}, "arm": {"both"}},
     "Only participants of a crossover trial can be in both arms."),
    ("PS2", {"sponsor": {"academic"}, "scope": {"multinational"}},
     "An academic sponsor does not run multinational trials."),
    ("PC8", {"design": {"crossover"}, "arm": {"experimental", "comparator"}},
     "In a crossover trial every participant goes through both arms."),
    ("P7", {"phase": {"IV"}, "control": {"placebo"}},
     "Post-marketing (phase IV) trials are not placebo-controlled."),
    # -- 24 rules
    ("PC9", {"control": {"active"}, "placebo": {"Yes"}},
     "An active-controlled trial gives no placebos."),
    ("CS3", {"status": {"withdrawn"}, "outcome": {"recorded"}},
     "A withdrawn participation has no final outcome."),
    ("PC10", {"control": {"none"}, "arm": {"comparator"}},
     "A trial without a control has no comparator arm."),
    ("P8", {"randomised": {"No"}, "blinding": {"double"}},
     "A non-randomised trial is not double-blind."),
    ("PC11", {"randomised": {"No"}, "arm": {"comparator"}},
     "A non-randomised trial does not assign participants to a comparator arm."),
    ("C6", {"age_group": {"adult"}, "consent": {"parental"}},
     "Adult participants consent for themselves."),
    # -- 30 rules
]

MAX_RULES = len(RULES)


def schema():
    return two_tables(PARENT_TABLE, PARENT_KEY, PARENT_DOMAINS, CHILD_TABLE, CHILD_KEY, CHILD_DOMAINS)


def rules(count=MAX_RULES):
    """The first ``count`` rules."""
    if not 1 <= count <= MAX_RULES:
        raise ValueError(f"count must be between 1 and {MAX_RULES}")
    return [Rule.of(atoms, name, meaning) for name, atoms, meaning in RULES[:count]]


def spec(count=18, parents=1000, children=5, seed=0, **overrides) -> DatasetSpec:
    """The benchmark baseline: uniform frequency tables and ``children`` per parent.

    The benchmark's errors are ``ErrorSpec(rate=error_pct / 100)``: random, in every rule attribute.
    """
    values = dict(
        schema=schema(),
        rules=rules(count),
        fanout=Fixed(children),
        parents=parents,
        seed=seed,
    )
    values.update(overrides)
    return DatasetSpec(**values)
