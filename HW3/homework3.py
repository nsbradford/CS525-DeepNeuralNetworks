# NOTE -- please do NOT put your name(s) in the Python code; instead, name the Python file
# itself to include your WPI username(s).

import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np
import math
import scipy
from sklearn.linear_model import LogisticRegression


def sigmoid(z):
  return 1.0 / (1 + np.exp(-1 * z))


def J_new(w, faces, labels, alpha=0.0):
    """ yhat = sigmoid(Xw)
        J(w) = -1/m * sum { y_j log yhat_j + (1-y_j)log(1-yhat_y) } + alpha/2 * w_t * w
        J(w) = -1/m * y * log(yhat) + (1-y) * log(1-yhat) + alpha/2 * w_t * w
    """
    m = labels.size
    predict = sigmoid(faces.dot(w))
    first = labels.dot(np.log(predict))
    second = (1 - labels).dot(np.log(1 - predict))
    reg = 0.5 * alpha * np.dot(w, w)
    return (-1.0/m) * (first + second) + reg


def gradJ_new(w, faces, labels, alpha=0.0):
    """ """
    x = faces.T
    n = labels.size
    predict = sigmoid(x.T.dot(w))
    return (1.0/n) * x.dot(predict - labels)


def method4 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 0#1e-3
    print('USE METHOD 4')
    print('Train 1-layer ANN with regularization alpha: ', alpha)

    # Use Xavier initialization, where Var(W) = 1/N_input
    sigma = np.sqrt(1.0 / trainingFaces.size) # = 24
    w  = np.random.randn(trainingFaces.shape[1]) * sigma
    epsilon = 5e-2 # 0.000001 # learning rate
    delta = 1e-7 # threshold 
    prevJ = J_new(w, trainingFaces, trainingLabels, alpha)
    diff = 1000
    count = 0
    while diff > delta:
        update = epsilon * gradJ_new(w, trainingFaces, trainingLabels, alpha)
        w = w - update
        newJ = J_new(w, trainingFaces, trainingLabels, alpha)
        diff = prevJ - newJ
        prevJ = newJ
        count += 1
        if count % 100 == 0: print('\t', newJ, diff) #print('...',) 
    return w


def whiten(data):
    """ L = evecs * evals^(-1/2) """
    assert data.shape == (2000,576), 'data.shape is ' + str(data.shape)
    constant = 1e-3
    cov = np.cov(data.T) + constant * np.eye(data.shape[1])
    # assert cov.shape == (576, 576), 'evals.shape is ' + str(evals.shape)
    evals, evecs = np.linalg.eigh(cov)
    # assert evals.shape == (576,), 'evals.shape is ' + str(evals.shape)
    evals_m = np.diagflat(np.float_power(evals, -0.5))
    # assert evals_m.shape == (576,576), 'evals_m.shape is ' + str(evals_m.shape)
    W = np.dot(evecs, evals_m)
    whitened = np.dot(data, W)
    return whitened


#==================================================================================================

def J(w, faces, labels, alpha=0.0):
    """ J = 0.5 * [Xw-y]^2 
        Update to use sigmoid and regularized cross-entropy loss function (2-class):
            J(w) = -(1/m) [ y_j * log(Xw) + (1 - y) * log(1 - Xw) ] + (alpha / 2) * w_T * w
    """
    # print([x.shape for x in [w, faces , labels]])
    inner = np.dot(faces, w) - labels
    return 0.5 * np.dot(inner, inner.T) + (alpha * np.dot(w, w))


def gradJ(w, faces, labels, alpha=0.0):
    """ gradJ = 0.5 * 2 * X.T(Xw - y) = X.T(Xw-y) + Alpha * w """
    inner = np.dot(faces, w) - labels
    return np.dot(faces.T, inner) + (alpha * w)


def gradientDescent (trainingFaces, trainingLabels, testingFaces, testingLabels, alpha=0.0):
    """ Random initialization from a standard normal distribution.
        Gradient descent is then applied until gain is < epsilon. 
    """
    print('Train 1-layer ANN with regularization alpha: ', alpha)
    w  = np.random.randn(trainingFaces.shape[1])
    epsilon = 2e-5 # 0.000001 # learning rate
    delta = 1e-3 #0.0001 
    prevJ = J(w, trainingFaces, trainingLabels, alpha)
    diff = 1000
    count = 0
    # print(diff, delta)
    while diff > delta:
        update = epsilon * gradJ(w, trainingFaces, trainingLabels, alpha)
        w = w - update
        newJ = J(w, trainingFaces, trainingLabels, alpha)
        diff = prevJ - newJ
        prevJ = newJ
        count += 1
        if count % 10 == 0: print('\t', count, newJ, diff) #print('...',) 
    return w

#==================================================================================================

def method1 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    """ Set gradient to 0 and solve. 
        w = (X.T * X)^-1 * (X.T * y)
        Use the fact that for Ax = b, x = A-1b -> np.solve(A, b)
    """
    # print([x.shape for x in [trainingFaces, trainingLabels, testingFaces, testingLabels]])
    leftTerm = np.dot(trainingFaces.T, trainingFaces)
    rightTerm = np.dot(trainingFaces.T, trainingLabels)
    return np.linalg.solve(leftTerm, rightTerm)


def method2 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels)


def method3 (trainingFaces, trainingLabels, testingFaces, testingLabels):
    alpha = 1e3
    return gradientDescent(trainingFaces, trainingLabels, testingFaces, testingLabels, alpha)


def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    print("Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha)))
    print("Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha)))


# Accesses the web camera, displays a window showing the face, and classifies smiles in real time
# Requires OpenCV.
def detectSmiles (w):
    # Given the image captured from the web camera, classify the smile
    def classifySmile (im, imGray, faceBox, w):
        # Extract face patch as vector
        face = imGray[faceBox[1]:faceBox[1]+faceBox[3], faceBox[0]:faceBox[0]+faceBox[2]]
        face = cv2.resize(face, (24, 24))
        face = (face - np.mean(face)) / np.std(face)  # Normalize
        face = np.reshape(face, face.shape[0]*face.shape[1])

        # Classify face patch
        yhat = w.dot(face)
        print(yhat)

        # Draw result as colored rectangle
        THICKNESS = 3
        green = 128 + (yhat - 0.5) * 255
        color = (0, green, 255 - green)
        pt1 = (faceBox[0], faceBox[1])
        pt2 = (faceBox[0]+faceBox[2], faceBox[1]+faceBox[3])
        cv2.rectangle(im, pt1, pt2, color, THICKNESS)

    # Starting video capture
    vc = cv2.VideoCapture()
    vc.open(0)
    path = "/Users/nicholasbradford/opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
    faceDetector = cv2.CascadeClassifier(path) # TODO update path

    while vc.grab():
        (tf,im) = vc.read()
        im = cv2.resize(im, (int(im.shape[1]/2), int(im.shape[0]/2)))  # Divide resolution by 2 for speed
        imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # print (imGray.shape)
        k = cv2.waitKey(30)
        if k >= 0 and chr(k) == 'q':
            print("quitting")
            break

        # # Detect faces
        faceBoxes = faceDetector.detectMultiScale(imGray)
        for faceBox in faceBoxes:
            classifySmile(im, imGray, faceBox, w)
        cv2.imshow("WebCam", im)

    cv2.destroyWindow("WebCam")
    vc.release()


if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    # Load data
    # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
    if ('trainingFaces' not in globals()):
        prefix = 'smile_data/'
        trainingFaces = np.load(prefix + "trainingFaces.npy")
        trainingLabels = np.load(prefix + "trainingLabels.npy")
        testingFaces = np.load(prefix + "testingFaces.npy")
        testingLabels = np.load(prefix + "testingLabels.npy")

    whitenedFaces = whiten(trainingFaces)
    # verify that the eigenvalues of hte covariance matrix of the transformed faces are all close to 1
    newCov = np.cov(whitenedFaces.T)
    newEvals, newEvecs = np.linalg.eigh(newCov)
    # print(newEvecs)

    # w1 = method1(trainingFaces, trainingLabels, testingFaces, testingLabels)
    w2 = method2(whitenedFaces, trainingLabels, testingFaces, testingLabels)
    # w3 = method3(trainingFaces, trainingLabels, testingFaces, testingLabels)
    # w4 = method4(trainingFaces, trainingLabels, testingFaces, testingLabels)

    sigma = np.sqrt(1.0 / trainingFaces.size) # = 24
    w  = np.random.randn(trainingFaces.shape[1]) * sigma
    C = 1e10 # inverse of alpha, the regularization rate
    model = LogisticRegression(C=C)
    point_to_check = w
    gradient_check = scipy.optimize.check_grad(J_new, gradJ_new, point_to_check, trainingFaces, trainingLabels)
    print(gradient_check)

    # for i, w in enumerate([ w1, w2, w3 , w4]):
    # for i, w in enumerate([ w2]):
    #     print('Costs for method', i+1, ':')
    #     reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)
    
    # detectSmiles(w2)  # Requires OpenCV