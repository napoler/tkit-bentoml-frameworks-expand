# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""
from sklearn import datasets
from sklearn import svm

from IrisClassifier import IrisClassifier

clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# Create a iris classifier service with the newly trained model
iris_classifier_service = IrisClassifier()
iris_classifier_service.pack("model", clf)

# Test invoking BentoService instance
iris_classifier_service.predict([[5.1, 3.5, 1.4, 0.2]])

# Start a dev model server to test out the API endpoint locally
iris_classifier_service.start_dev_server()

# Stop the dev model server
iris_classifier_service.stop_dev_server()

# Save the entire prediction service to a BentoML bundle
saved_path = iris_classifier_service.save()
if __name__ == '__main__':
    pass
