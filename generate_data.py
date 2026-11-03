"""Generate a synthetic past-student dataset (students.csv).

Pass/fail is driven by attendance, assignment marks and previous exam marks,
plus noise, so the model has a real (but imperfect) pattern to learn.
"""
import numpy as np
import pandas as pd


def generate(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    attendance = np.clip(rng.normal(70, 18, n), 15, 100)
    assignment = np.clip(rng.normal(60, 20, n), 0, 100)
    prev_exam = np.clip(rng.normal(55, 20, n), 0, 100)

    # Weighted score + noise -> probability of passing
    score = 0.30 * attendance + 0.30 * assignment + 0.40 * prev_exam
    score += rng.normal(0, 4, n)
    prob_pass = 1 / (1 + np.exp(-(score - 60) / 4))
    passed = (rng.random(n) < prob_pass).astype(int)

    return pd.DataFrame(
        {
            "student_id": [f"S{i:04d}" for i in range(1, n + 1)],
            "attendance": attendance.round(1),
            "assignment_marks": assignment.round(1),
            "previous_exam_marks": prev_exam.round(1),
            "passed": passed,  # 1 = passed, 0 = failed
        }
    )


if __name__ == "__main__":
    df = generate()
    df.to_csv("students.csv", index=False)
    print(f"Saved students.csv ({len(df)} rows, pass rate {df.passed.mean():.0%})")
