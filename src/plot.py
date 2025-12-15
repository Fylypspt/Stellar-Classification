import matplotlib.pyplot as plt

def graph(y_pred, y_test, model, X_test):
    plt.figure(figsize=(9, 7))

    correct_mask = y_pred == y_test
    mis_mask = ~correct_mask

    for cls in model.classes_:
        idx = (y_test == cls) & correct_mask
        plt.scatter(
            X_test.loc[idx, "u"],
            X_test.loc[idx, "redshift"],
            alpha=0.5,
            s=35,
            label=f"{cls} (correct)"
        )

    plt.scatter(
        X_test.loc[mis_mask, "u"],
        X_test.loc[mis_mask, "redshift"],
        c="red",
        marker="x",
        s=70,
        linewidths=1.5,
        label="Misclassified"
    )

    plt.xlabel("u magnitude")
    plt.ylabel("Redshift")
    plt.title("Star Classification — u vs Redshift")
    plt.legend(frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()