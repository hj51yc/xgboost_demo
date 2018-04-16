#!encoding=utf8

from __future__ import division
import numpy as np
import sys, os
import xgboost

def load_xgb_data(filename):
    data = np.loadtxt(filename, delimiter=',', 
        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x) - 1})
    row, col = data.shape
    split_index = int(row * 0.7)
    train_data = data[:split_index, :]
    test_data = data[split_index:, :]
    train_x = train_data[:, :33]
    train_y = train_data[:, 34]

    test_x = test_data[:, :33]
    test_y = test_data[:, 34]

    xgb_train_data = xgboost.DMatrix(train_x, label=train_y)
    xgb_test_data = xgboost.DMatrix(test_x, label=test_y)
    return xgb_train_data, xgb_test_data, train_x, train_y, test_x, test_y


def train():
    filename = "./dermatology.data"
    trainDMatrix, testDMatrix, train_x, train_y, test_x, test_y = load_xgb_data(filename)
    params = {
        "max_depth": 4,
        "eta": 0.2, ##learning rate
        'objective': 'multi:softmax',
        'silent': 1,
        'nthread': 4,
        'num_class': 6,
        'colsample_bytree': 0.6,
        'lambda': 1.5,
        'alpha': 1.0,
        'colsample_bylevel': 0.6,
        'gamma' : 0.05
    }
    boost_round = 100
    watchlist = [(trainDMatrix, 'train'), (testDMatrix, 'test')]
    xgb_model = xgboost.train(params, trainDMatrix, boost_round, watchlist)
    pred = xgb_model.predict(testDMatrix, pred_leaf=True)
    np.savetxt("pred.log", pred, fmt="%.2f")
    #pred = xgb_model.predict(testDMatrix, pred_contribs=True) #not right
    #pred = xgb_model.predict(testDMatrix)
    #error_rate = np.sum(pred != test_y) / test_y.shape[0]
    #print 'softmax error rate:', error_rate

    #params['objective'] = 'multi:softprob'
    #bst = xgboost.train(params, trainDMatrix, boost_round, watchlist)
    #pred_prob = bst.predict(testDMatrix).reshape(test_y.shape[0], 6)
    #pred_label = np.argmax(pred_prob, axis=1)
    #error_rate = np.sum(pred != test_y) / test_y.shape[0]
    #print 'softprob error rate:', error_rate



def grid_search_train():
    from xgboost import XGBClassifier
    from sklearn.grid_search import GridSearchCV
    filename = "./dermatology.data"
    trainDMatrix, testDMatrix, train_x, train_y, test_x, test_y = load_xgb_data(filename)
    params = {
        "max_depth": [2, 3, 4],
        "learning_rate": [0.02, 0.05, 0.1, 0.2], ##learning rate
        #"colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0]
        "colsample_bylevel": [0.6, 0.8,  1.0],
        "reg_alpha": [0.3, 0.6, 0.8, 1.0],
        "reg_lambda": [0.5, 1, 1.5, 2],
        #'objective': 'multi:softmax',
        #'silent': 1,
        #'nthread': 4,
        #'num_class': 6
    }
    boost_round = 20
    watchlist = [(trainDMatrix, 'train'), (testDMatrix, 'test')]
    xgb_model = XGBClassifier(
            #num_class = 6, ##when use grid_search, it will count uniq y len
            objective="multi:softmax",
            silent = 1,
            nthread = 4,
            gamma = 0.05,
            subsample=0.6,
            colsample_bytree = 0.9,
            reg_lambda = 1, #L2 norm
            reg_alpha = 0.1, #L1 norm
            #scale_pos_weight = 1 # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
            max_delta_step = 0, #最大增量步长，我们允许每个树的权重估计
            n_estimators = boost_round,
            #eval_metric= 'auc' #rmse, mae, logloss, error, merror, mlogloss , auc
        )
    #clf = GridSearchCV(xgb_model, params, scoring='roc_auc')
    clf = GridSearchCV(xgb_model, params, scoring='accuracy')
    clf.fit(train_x, train_y)
    print clf.best_params_
    print clf.best_score_




if __name__ == "__main__":
    train()
    #grid_search_train()
    
