import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import DecisionTreeClassifier


def convert_descriptor_to_histogram(descriptors, vocab_model, normalize=True) -> np.ndarray:
    features_words = vocab_model.predict(descriptors)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram

def apply_feature_transform(
        data: np.ndarray,
        feature_detector_descriptor,
        vocab_model
) -> np.ndarray:
    data_transformed = []
    for image in data:
        keypoints, image_descriptors = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = convert_descriptor_to_histogram(image_descriptors, vocab_model)
        data_transformed.append(bow_features_histogram)
    return np.asarray(data_transformed)


np.random.seed(42)
images = []
labels = []

classified_images = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: []

}

# class dir i image_path to obiekty typu Path więc w odwołaniu cv2.imread musimy konwertowac na stringa
for class_id , class_dir in enumerate(sorted(Path('/content/drive/MyDrive/Colab Notebooks/Pliki/data_combined').iterdir())):
    for image_path in class_dir.iterdir():
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        classified_images[class_id].append(image)
        images.append(image)
        labels.append(class_id)


train_images = images
train_labels = labels

#feature_detector_descriptor = cv2.BRISK_create()
#feature_detector_descriptor = cv2.ORB_create(nfeatures = 5000)
#feature_detector_descriptor = cv2.xfeatures2d.SURF_create()#extended=True,upright=True)
#feature_detector_descriptor = cv2.ORB_create(nfeatures = 5000)
feature_detector_descriptor = cv2.xfeatures2d.SIFT_create()

train_descriptors = [descriptor for descriptor in feature_detector_descriptor.detectAndCompute(image,None)[1]
                     for image in train_images]

print('Descriptors: ', len(train_descriptors))

NB_WORDS = 150
#kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42)
kmeans = cluster.KMeans(n_clusters=NB_WORDS, random_state=42,n_init= 40,tol=0.00001,algorithm = 'elkan')
kmeans.fit(train_descriptors)

file = open('vocab_model.p', 'wb')
pickle.dump(kmeans, file)
file.close()

with Path('vocab_model.p').open('rb') as vocab_file:
  kmeans = pickle.load(vocab_file)



X_train = apply_feature_transform(train_images, feature_detector_descriptor, kmeans)
y_train = train_labels

# classifier = RandomForestClassifier(n_estimators=1000,max_features='auto',criterion='gini',random_state=42)
clf_svm = svm.SVC()
clf_svm.fit(X_train, y_train)

print(clf_svm.score(X_train, y_train))
param_grid = {
    # 'max_depth' : [1,3,5,10,30],
    # 'n_estimators' : [1,5,10,50,100,200,500],
    # 'criterion' : ['gini','entropy']
    'C': [0.1, 0.5, 1, 3, 5, 10, 100, 1000],
    'coef0': [1.0, 2.0, 3.0],
    'kernel': ['poly']
}
k_fold = StratifiedKFold(n_splits=10)
grid_search = GridSearchCV(clf_svm, param_grid, cv=k_fold)
grid_search.fit(X_train, y_train)

file = open('clf.p', 'wb')
pickle.dump(grid_search, file)
file.close()

with Path('clf.p').open('rb') as classifier_file:  # Don't change the path here
    clf = pickle.load(classifier_file)


print(grid_search.score(X_train, y_train))
print(grid_search.score(X_test, y_test))
print(grid_search.best_params_)