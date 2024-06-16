import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

alpha               = 0.01    # --- Learning rate
numIter             = 2500    # --- Number of gradient descent iterations
batchSize           = 64      # --- Batch size for the train data

numFeatures         = 2

#xDataset        = np.array([[x[0], x[3]] for x in irisDataset.data])
#yDataset        = np.array([1 if y == 0 else -1 for y in irisDataset.target])

xDataset        = np.array(data[:,[3,4]])
yDataset        = data[:,[6]]

xDataset   = xDataset.astype('float32')
yDataset   = yDataset.astype('float32')


#print(xDataset)
#print(yDataset)
trainIndices    = np.random.choice(len(xDataset), round(len(xDataset) * 0.9), replace = False)
testIndices     = np.array(list(set(range(len(xDataset))) - set(trainIndices)))
xTrain          = xDataset[trainIndices]
xTest           = xDataset[testIndices]
yTrain          = yDataset[trainIndices]
yTest           = yDataset[testIndices]
randomInitWeights = tf.initializers.Zeros()

w = tf.Variable(randomInitWeights([numFeatures, 1], dtype = np.float32))
w0 = tf.Variable(randomInitWeights([1, 1], dtype = np.float32))

def costFunction(xData, yTarget):
  #model               = tf.subtract(tf.matmul(xData, w), b)
  model               = tf.add(tf.matmul(xData, w), w0)
  regularizationTerm  = tf.reduce_sum(tf.square(w))
  classificationTerm  = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model, yTarget))))
  return tf.add(classificationTerm, tf.multiply(alpha, regularizationTerm))
optimizer = tf.optimizers.SGD(alpha)

# --- Optimization step
def optimizationStep(features, classes):
    # --- Uses GradientTape for automatic differentiation
    with tf.GradientTape() as g:
        costFunctionValue = costFunction(features, classes)

    # --- Compute gradients
    gradients = g.gradient(costFunctionValue, [w, w0])
    
    # --- Update the unknowns W and w0
    optimizer.apply_gradients(zip(gradients, [w, w0]))

def predictionAccuracy(xData, yTarget):
  
  #model        = tf.subtract(tf.matmul(xData, A), w0)
  model        = tf.add(tf.matmul(xData, w), w0)
  prediction   = tf.reshape(tf.sign(model), [-1])
  accuracy     = tf.reduce_mean(tf.cast(tf.equal(prediction, yTarget), tf.float32))

  return accuracy

costFunctionVec   = []
testAccuracyVec   = []

# --- Optimization loop
for i in range(numIter):
  indexBatch    = np.random.choice(len(xTrain), size = batchSize)
  xBatch        = xTrain[indexBatch]
  yBatch        = np.transpose([yTrain[indexBatch]])
  
  optimizationStep(xBatch, yBatch)

  currentCost             = costFunction(xBatch, yBatch)
  currentTestPrediction   = predictionAccuracy(xTest, yTest)
  #print('Cost function = {}; accuracy = {}'.format(currentCost, currentTestPrediction))

  costFunctionVec.append(currentCost)
  testAccuracyVec.append(currentTestPrediction)

# --- Computing the dividing line
m             = -w[1] / w[0]
x0            = -w0    / w[0]
xSep          = [x[1] for x in xDataset]
ySep          = [m * x + x0 for x in xSep]

setosaX       = [d[1] for i, d in enumerate(xDataset) if yDataset[i] ==  1]
setosaY       = [d[0] for i, d in enumerate(xDataset) if yDataset[i] ==  1]
notSetosaX    = [d[1] for i, d in enumerate(xDataset) if yDataset[i] == -1]
notSetosaY    = [d[0] for i, d in enumerate(xDataset) if yDataset[i] == -1]

plt.plot(setosaX,    setosaY,     'ro', label = 'radical')
plt.plot(notSetosaX, notSetosaY,  'g*', label = 'seed')
plt.plot(np.reshape(xSep, (len(xSep), 1)), np.reshape(ySep, (len(ySep), 1)), 'b-', label='Dividing line')
plt.xlim([10, 255])
plt.ylim([10, 255])
plt.title('Seed And Radical')
plt.xlabel('H')
plt.ylabel('S')
plt.legend(loc = 'lower right')
plt.show()

plt.plot(testAccuracyVec, 'r--', label = 'Test accuracy')
plt.title('Test set accuracy')
plt.xlabel('Iteration')
plt.ylabel('accuracy')
plt.legend(loc = 'lower right')
plt.show()

plt.plot(costFunctionVec, 'k-')
plt.title('Cost functional per iteration')
plt.xlabel('Iteration')
plt.ylabel('Cost functional')
plt.show()
