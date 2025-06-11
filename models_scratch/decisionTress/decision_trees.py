import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self, featureIndex=None, threshold=None, leftChild=None, rightChild=None, *, predictedClass=None):
        self.featureIndex = featureIndex
        self.threshold = threshold
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.predictedClass = predictedClass

    def isLeaf(self):
        return self.predictedClass is not None


class DecisionTreeClassifier:
    def __init__(self, minSamplesSplit=2, maxDepth=100, maxFeatures=None):
        self.minSamplesSplit = minSamplesSplit
        self.maxDepth = maxDepth
        self.maxFeatures = maxFeatures
        self.rootNode = None

    def fit(self, features, labels):
        numTotalFeatures = features.shape[1]
        self.numSelectedFeatures = numTotalFeatures if self.maxFeatures is None else min(self.maxFeatures, numTotalFeatures)
        self.rootNode = self._buildTree(features, labels)

    def _buildTree(self, features, labels, currentDepth=0):
        numSamples, numFeatures = features.shape
        numClasses = len(np.unique(labels))

        if (currentDepth >= self.maxDepth or numClasses == 1 or numSamples < self.minSamplesSplit):
            leafClass = self._majorityVote(labels)
            return TreeNode(predictedClass=leafClass)

        selectedFeatureIndices = np.random.choice(numFeatures, self.numSelectedFeatures, replace=False)

        bestFeatureIndex, bestThreshold = self._findBestSplit(features, labels, selectedFeatureIndices)

        leftIndices, rightIndices = self._splitDataset(features[:, bestFeatureIndex], bestThreshold)

        leftSubtree = self._buildTree(features[leftIndices, :], labels[leftIndices], currentDepth + 1)
        rightSubtree = self._buildTree(features[rightIndices, :], labels[rightIndices], currentDepth + 1)

        return TreeNode(featureIndex=bestFeatureIndex, threshold=bestThreshold,
                        leftChild=leftSubtree, rightChild=rightSubtree)

    def _findBestSplit(self, features, labels, featureIndices):
        bestGain = -1
        bestFeatureIndex = None
        bestThreshold = None

        for featureIndex in featureIndices:
            featureColumn = features[:, featureIndex]
            thresholds = np.unique(featureColumn)

            for threshold in thresholds:
                gain = self._informationGain(labels, featureColumn, threshold)

                if gain > bestGain:
                    bestGain = gain
                    bestFeatureIndex = featureIndex
                    bestThreshold = threshold

        return bestFeatureIndex, bestThreshold

    def _informationGain(self, labels, featureColumn, threshold):
        parentEntropy = self._calculateEntropy(labels)

        leftIndices, rightIndices = self._splitDataset(featureColumn, threshold)

        if len(leftIndices) == 0 or len(rightIndices) == 0:
            return 0

        total = len(labels)
        leftWeight = len(leftIndices) / total
        rightWeight = len(rightIndices) / total

        leftEntropy = self._calculateEntropy(labels[leftIndices])
        rightEntropy = self._calculateEntropy(labels[rightIndices])

        weightedEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy

        return parentEntropy - weightedEntropy

    def _splitDataset(self, featureColumn, threshold):
        leftIndices = np.argwhere(featureColumn <= threshold).flatten()
        rightIndices = np.argwhere(featureColumn > threshold).flatten()
        return leftIndices, rightIndices

    def _calculateEntropy(self, labels):
        classCounts = np.bincount(labels)
        probabilities = classCounts / len(labels)
        return -np.sum([p * np.log(p) for p in probabilities if p > 0])

    def _majorityVote(self, labels):
        labelCounter = Counter(labels)
        mostCommonLabel = labelCounter.most_common(1)[0][0]
        return mostCommonLabel

    def predict(self, features):
        return np.array([self._traverseTree(sample, self.rootNode) for sample in features])

    def _traverseTree(self, sample, node):
        if node.isLeaf():
            return node.predictedClass

        if sample[node.featureIndex] <= node.threshold:
            return self._traverseTree(sample, node.leftChild)
        return self._traverseTree(sample, node.rightChild)
