**Problem Statement:**

This project addresses the Maintenance Programming of Electrical Power Systems, which involves solving a challenging problem classified as an NP-complete problem.

**Problem Description:**

In an electrical power system, preventive maintenance is required, which involves shutting down machines. The machine shutdowns pose safety risks to the entire system. The objective is to determine a sequence for stopping machines within a given period (e.g., 1 year) to maximize system safety. Safety is measured by the net reserve of power (Pl), calculated using Equation 1:

Pl = Pt - Pp - Pd

Where:
- Pt: Total installed power
- Pp: Power lost per stop
- Pd: Maximum power demand

**Constraints:**

1. Maintenance of any machine starts at the beginning of a break and ends at the end of the same interval or the next interval if there are multiple stops. Maintenance cannot be interrupted or rescheduled.

2. The net power reserve (Pl, as per Equation 1) must be positive and not zero during any intermission.

**Optimization Criterion:**

The optimization goal is to find the maximum value of Pl for each interval while adhering to the specified constraints.
