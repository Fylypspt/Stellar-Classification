This project classifies stars, galaxies, and quasars using only basic SDSS Photometric data (u, g, r, i, z and redshift).

A Random Forest is used because it handles noisy tabular data well and is easy to reason about. The model is trained once and reused so results stay consistent and iteration is fast.

The project focuses on robustness. It deliberately simulates data drift to test how performance degrades when observations are slightly off, and it allows the model to output “UNCERTAIN” instead of forcing low-confidence predictions.

Evaluation looks at what actually goes wrong, not just accuracy, with special attention to confusion between galaxies and quasars.
