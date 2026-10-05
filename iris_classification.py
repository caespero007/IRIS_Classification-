from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# -----------------------------------
# 1. Load the Iris dataset
# -----------------------------------

iris = load_iris()

X = iris.data
y = iris.target


print("\nDataset Information")
print("-------------------")

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFeature names:")
for feature in iris.feature_names:
    print("-", feature)

print("\nTarget names:")
for target in iris.target_names:
    print("-", target)

# -----------------------------------
# 2. Split the dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# 3. Create the model
# -----------------------------------

model = LogisticRegression(max_iter=200)


# -----------------------------------
# 4. Train the model
# -----------------------------------

model.fit(X_train, y_train)


# -----------------------------------
# 5. Make predictions
# -----------------------------------

predictions = model.predict(X_test)


# -----------------------------------
# 6. Calculate accuracy
# -----------------------------------

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")


# -----------------------------------
# 7. Confusion Matrix
# -----------------------------------

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()

plt.title("Iris Flower Classification - Confusion Matrix")

plt.savefig("confusion_matrix.png")

plt.show()


# -----------------------------------
# 8. Scatter Plot
# -----------------------------------

plt.figure(figsize=(8, 5))

for i in range(3):

    plt.scatter(
        X[y == i, 2],
        X[y == i, 3],
        label=iris.target_names[i]
    )

plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris Flower Dataset")

plt.legend()

plt.savefig("iris_scatter_plot.png")

plt.show()


# -----------------------------------
# 9. Histogram
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.hist(X[:, 0], bins=10)

plt.xlabel("Sepal Length")
plt.ylabel("Number of Flowers")

plt.title("Distribution of Sepal Length")

plt.savefig("sepal_length_histogram.png")

plt.show()
