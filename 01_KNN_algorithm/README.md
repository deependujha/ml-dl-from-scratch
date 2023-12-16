# KNN algorithm

![knn algorithm](../assets/01_KNN/KNN.png)

### Note 🚫:
Current implement creates a whole new array just to track the distances of a point to all the other points.

```py
# _predict method of KNN class
distances = [calculate_distance(x_train, x) for x_train in self.X]
```

- Space complexity: O(n)
- But, we only need **`k sized array`** and we can  use it to store the tuple **`(distances, class it belongs to)`**.

