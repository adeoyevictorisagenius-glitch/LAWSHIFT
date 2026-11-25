"""Generate the SYNTHETIC sample CSV files in sample_data/ (reproducible, seed fixed).

Each file is produced from a known law plus measurement noise, with a distractor column
that has no effect. LAWSHIFT never sees these laws; they are only used here to make data.
    python make_sample_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(2024)
n = 200


def noisy(y, frac):
    return y + rng.normal(0, frac * np.std(y), len(y))


# 1. Pendulum: T = 2*pi*sqrt(L/g)   (mass is a distractor)
L = rng.uniform(0.2, 2.0, n)
pd.DataFrame({"Length (m)": L.round(4), "Mass (kg)": rng.uniform(0.1, 5, n).round(3),
              "Period (s)": noisy(2 * np.pi * np.sqrt(L / 9.81), 0.01).round(5)}).to_csv(OUT / "pendulum_period.csv", index=False)

# 2. Ideal gas: P = n*R*T/V   (kPa, with V in litres)
T, V, mol = rng.uniform(250, 500, n), rng.uniform(5, 50, n), rng.uniform(0.5, 3, n)
pd.DataFrame({"Temperature (K)": T.round(2), "Volume (L)": V.round(3), "Amount (mol)": mol.round(3),
              "Lab humidity (%)": rng.uniform(20, 80, n).round(1),
              "Pressure (kPa)": noisy(8.314 * mol * T / V, 0.02).round(3)}).to_csv(OUT / "ideal_gas.csv", index=False)

# 3. Mass on a spring: x = 2*cos(3 t)
t = rng.uniform(0, 6, n)
pd.DataFrame({"Time (s)": t.round(4), "Displacement (m)": noisy(2 * np.cos(3 * t), 0.02).round(5)}).to_csv(
    OUT / "spring_oscillation.csv", index=False)

# 4. Electrical power: P = V^2 / R   (room temperature is a distractor)
Vv, R = rng.uniform(1, 24, n), rng.uniform(10, 200, n)
pd.DataFrame({"Voltage (V)": Vv.round(3), "Resistance (ohm)": R.round(2), "Room temp (C)": rng.uniform(15, 30, n).round(1),
              "Power (W)": noisy(Vv ** 2 / R, 0.02).round(5)}).to_csv(OUT / "electrical_power.csv", index=False)
# ---- everyday (non-physics) samples for the Everyday Data Lab. Appended after the physics files so those stay unchanged.
m = 240

# 5. Coffee shop: cups sold per hour. Price lowers sales exponentially, staff has diminishing returns, and there is a
#    sweet-spot temperature (hot drinks sell best on cool days). Shift number is a distractor.
temp = rng.uniform(5, 35, m)
price = rng.uniform(2.5, 6.0, m)
traffic = np.clip(rng.normal(60, 15, m), 15, None)
staff = rng.integers(2, 9, m).astype(float)
cups = (traffic * 0.55 * np.exp(-0.38 * (price - 2.5)) * (1 - 0.5 * ((temp - 12) / 25) ** 2)
        * (1 - np.exp(-0.55 * staff)))
pd.DataFrame({"Temperature (C)": temp.round(1), "Price (USD)": price.round(2), "Foot traffic (people/h)": traffic.round(0),
              "Staff on shift": staff.astype(int), "Shift number": rng.integers(1, 4, m),
              "Cups sold (per hour)": noisy(cups, 0.04).round(1)}).to_csv(OUT / "coffee_shop.csv", index=False)

# 6. Study habits: exam score. Studying has diminishing returns, sleep has a sweet spot near 8 hours, screen time hurts.
#    Screen time is deliberately right-skewed (log-normal); coffee is a distractor.
study = np.clip(rng.normal(3.0, 1.0, m), 0.2, None)
sleep = np.clip(rng.normal(7.0, 1.1, m), 3.5, 10.5)
screen = rng.lognormal(0.9, 0.55, m)
score = 38 + 40 * (1 - np.exp(-0.6 * study)) - 2.0 * (sleep - 8) ** 2 - 1.0 * screen
pd.DataFrame({"Study hours (per day)": study.round(2), "Sleep (hours)": sleep.round(2), "Screen time (hours)": screen.round(2),
              "Coffee (cups)": rng.integers(0, 6, m),
              "Exam score": np.clip(noisy(score, 0.05), 0, 100).round(1)}).to_csv(OUT / "study_habits.csv", index=False)
print("wrote", sorted(p.name for p in OUT.glob("*.csv")))
