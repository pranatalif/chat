# https: // towardsdatascience.com/anomaly-detection-with-local-outlier-factor-lof-d91e41df10f2
# data preparation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pandas as pd
from pandas import read_csv
import numpy as np
# data visualzation
import matplotlib.pyplot as plt
import seaborn as sns
# outlier/anomaly detection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from custom_scripts.evaluate_performance import evaluate_performance

# url = '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/pca_array.csv'
url = '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/Outlier-detection-dataset.csv'

data = read_csv(url)

# df = pd.DataFrame(data, columns=[
#                   "x", "y"])
df = pd.DataFrame(data, columns=[
    "Cpu", "Req-to-Hls", "Req-to-Kc", "Http-200", "Http-204", "Http-302", "Rcvd", "Sent", "Rss-mem", "Used-mem", "Avail-mem", "PC1", "PC2"])
target = pd.DataFrame(data, columns=[
    "target"])


scaler1 = StandardScaler()
scaler1.fit(data)
feature_scaled = scaler1.transform(data)

pca1 = PCA(n_components=4)
pca1.fit(feature_scaled)
feature_scaled_pca = pca1.transform(feature_scaled)
print("shape of the scaled and 'PCA'ed features: ", np.shape(feature_scaled_pca))
res = pd.DataFrame(pca1.transform(feature_scaled))

feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))

print("Variance Ratio of the 4 Principal Components Ananlysis: ", feat_var_rat)

chaos_target_list = data.target.tolist()
feature_scaled_pca_X0 = feature_scaled_pca[:, 0]
feature_scaled_pca_X1 = feature_scaled_pca[:, 1]

# print("feature_scaled_pca_X0: ", feature_scaled_pca_X0)
# print("feature_scaled_pca_X1: ", feature_scaled_pca_X1)

# pca_array = np.asarray([feature_scaled_pca_X0, feature_scaled_pca_X1])
# print("gabungan: ", type(pca_array))
# np.savetxt("pca_outlier_detection.csv", pca_array.transpose(), delimiter=",")

# labels = chaos_target_list
# colordict = {1: 'lime', 2: 'purple', 3: 'red',
#              4: 'blue'}
# piclabel = {1: 'Baseline (E$_B$)', 2: 'Recofiguration (E$_R$)',
#             3: 'Perturbation (E$_P$)', 4: r'Reconfiguration, Perturbation (E$_{RP}$)'}
# markers = {1: 'o', 2: '*', 3: 'd', 4: '+'}
# alphas = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}

# fig = plt.figure(figsize=(16, 7))
# for l in np.unique(labels):
#     ix = np.where(labels == l)
#     plt.scatter(feature_scaled_pca_X0[ix], feature_scaled_pca_X1[ix], c=colordict[l],
#                 label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])
#     # print("feature_scaled_pca_X0[ix]: ", feature_scaled_pca_X0[ix])
#     # print("type: ", type(feature_scaled_pca_X0[ix]))
#     # print("feature_scaled_pca_X1[ix]: ", feature_scaled_pca_X1[ix])
#     # print("size: ", feature_scaled_pca_X1[ix].shape)
# plt.xlabel("1st Principal Component", fontsize=15)
# plt.ylabel("2nd Principal Component", fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)

# plt.legend(fontsize=15)


# plt.show()
url1 = '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/pca_outlier_detection.csv'

data = read_csv(url1)

df = pd.DataFrame(data, columns=[
                  "x", "y"])
# df1 = pd.DataFrame(data, columns=[
#     "Cpu", "Req-to-Hls", "Req-to-Kc", "Http-200", "Http-204", "Http-302", "Rcvd", "Sent", "Rss-mem", "Used-mem", "Avail-mem", "PC1", "PC2"])
target1 = pd.DataFrame(data, columns=[
    "target"])

X_train, X_test, Y_train, Y_test = train_test_split(
    data, data.target, test_size=0.25)

# np.savetxt("X_train_outlier.csv", X_train, delimiter=",")
# np.savetxt("X_test_outlier.csv", X_test, delimiter=",")
# np.savetxt("Y_train_outlier.csv", Y_train, delimiter=",")
# np.savetxt("Y_test_outlier.csv", Y_test, delimiter=",")
# X_train = X_train.sort_index()
# X_test = X_test.sort_index()


# print("X_train: ", X_train)
# print("X_test: ", X_test)
# print("data: ", data)
# print("Df: ", df)
# print("data: ", target)
# print("=============================")
# print("X_train type: ", type(X_train))
# print("X_train shape: ", X_train.shape)
# print("X_test: ", X_test)
# print("X_test type: ", type(X_test))
# print("X_test shape: ", X_test.shape)
# print("Y_train: ", Y_train)
# print("Y_train type: ", type(Y_train))
# print("Y_train shape: ", Y_train.shape)
# print("Y_test: ", Y_test)
# print("Y_test type: ", type(Y_test))
# print("Y_test shape: ", Y_test.shape)

url_x_train = '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/X_train_outlier.csv'
url_y_train = '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/Y_train_outlier.csv'
url_x_test = '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/X_test_outlier.csv'
url_y_test = '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/First Paper/results/pca/Y_test_outlier.csv'
# data_X_train = read_csv(url_x_train)
data_X_train, data_X_test, data_Y_train, data_Y_test = read_csv(url_x_train), read_csv(
    url_x_test), read_csv(url_y_train), read_csv(url_y_test)
# df_X_train = pd.DataFrame(data_X_train, columns=["x", "y"])
df_X_train, df_X_test = pd.DataFrame(data_X_train, columns=[
    "x", "y"]), pd.DataFrame(data_X_test, columns=[
        "x", "y"])
# target_X_train = pd.DataFrame(df_X_train, columns=["target"])
target_X_train, target_X_test = pd.DataFrame(df_X_train, columns=[
    "target"]), pd.DataFrame(df_X_test, columns=[
        "target"])

baseline_X_train, reconfiguration_X_train, perturbation_X_train, mixed_X_train = df_X_train[data_X_train["target"] ==
                                                                                            1], df_X_train[data_X_train["target"] == 2], df_X_train[data_X_train["target"] == 3], df_X_train[data_X_train["target"] == 4]
baseline_X_test, reconfiguration_X_test, perturbation_X_test, mixed_X_test = df_X_test[data_X_test["target"] ==
                                                                                       1], df_X_test[data_X_test["target"] == 2], df_X_test[data_X_test["target"] == 3], df_X_test[data_X_test["target"] == 4]
print("============ TRAINING SIZE ============")
print("Size of baseline: ", len(baseline_X_train))
print("Size of reconfiguration: ", len(reconfiguration_X_train))
print("Size of perturbation: ", len(perturbation_X_train))
print("Size of mixed: ", len(mixed_X_train))
print("============ TESTING SIZE ============")
print("Size of baseline: ", len(baseline_X_test))
print("Size of reconfiguration: ", len(reconfiguration_X_test))
print("Size of perturbation: ", len(perturbation_X_test))
print("Size of mixed: ", len(mixed_X_test))

# ============ ALGORITHMS ============
# model1 = LinearRegression()
model2 = IsolationForest(
    n_estimators=200, max_samples=200, contamination=0.1, random_state=100)
# model2 = IsolationForest()
model3 = OneClassSVM(kernel='linear', gamma='auto', nu=0.1)  # fix
# model3 = OneClassSVM(kernel='poly', gamma='scale', nu=0.01)
# model4 = LocalOutlierFactor(
#     n_neighbors=300, metric="euclidean", contamination=0.1)
model4 = LocalOutlierFactor(n_neighbors=200, algorithm="brute",
                            leaf_size=200, contamination=0.1)  # fix
# model5 = DBSCAN()
model6 = EllipticEnvelope(
    contamination=0.10, random_state=100, support_fraction=0.1)  # fix

# model fitting outlier detection
# print("====== OUTLIER DETECTION =======")
X_train_pred2, X_test_pred2 = model2.fit_predict(
    df_X_train), model2.fit_predict(df_X_test)
X_train_pred3, X_test_pred3 = model3.fit_predict(
    df_X_train), model3.fit_predict(df_X_test)
X_train_pred4, X_test_pred4 = model4.fit_predict(
    df_X_train), model4.fit_predict(df_X_test)
# y_pred5 = model5.fit_predict(df)
X_train_pred6, X_test_pred6 = model6.fit_predict(
    df_X_train), model6.fit_predict(df_X_test)


# print("====== NOVELTY DETECTION =======")
# model2.fit(df_X_train), model2.fit(df_X_test)
# novelty_X_train_pred2, novelty_X_test_pred2 = model2.predict(
#     df_X_train), model2.predict(df_X_test)
model2.fit(df_X_train)
novelty_X_train_pred2 = model2.predict(df_X_test)


# print("X_train_pred2 bbbbb: ", X_train_pred2)
# print("novelty_X_train_pred2 bbbbb: ", novelty_X_train_pred2)

# with np.printoptions(threshold=np.inf):
# print("X_train_pred2 shape: ", X_train_pred2.size)
# print("X_train_pred3 shape: ", X_train_pred3)
# print("X_train_pred4 shape: ", X_train_pred4.size)
# print("X_train_pred6 shape: ", X_train_pred6)
# np.savetxt("pred2.csv", X_train_pred2, delimiter=" ")
# np.savetxt("pred3.csv", X_train_pred3, delimiter=" ")
# np.savetxt("pred4.csv", X_train_pred4, delimiter=" ")
# np.savetxt("pred6.csv", X_train_pred6, delimiter=" ")
# print("X_test_pred2 shape: ", X_test_pred2)
# print("X_test_pred3 shape: ", X_test_pred3.size)
# print("X_test_pred4 shape: ", X_test_pred4)
# print("X_test_pred6 shape: ", X_test_pred6.size)
# np.savetxt("pred2test.csv", X_test_pred2, delimiter=" ")
# np.savetxt("pred3test.csv", X_test_pred3, delimiter=" ")
# np.savetxt("pred4test.csv", X_test_pred4, delimiter=" ")
# np.savetxt("pred6test.csv", X_test_pred6, delimiter=" ")

# np.savetxt("pred2train_novelty.csv", novelty_X_train_pred2, delimiter=" ")
# np.savetxt("pred2test_novelty.csv", novelty_X_test_pred2, delimiter=" ")

# filter outlier index
# negative values are outliers and positives inliers
X_train_outlier_index2, X_test_outlier_index2 = np.where(
    X_train_pred2 == -1), np.where(X_test_pred2 == -1)
# with np.printoptions(threshold=np.inf):
X_train_outlier_index3, X_test_outlier_index3 = np.where(
    X_train_pred3 == -1), np.where(X_test_pred3 == -1)
# print("X_train_outlier_index3 outlier: ", X_train_outlier_index3)
X_train_outlier_index4, X_test_outlier_index4 = np.where(
    X_train_pred4 == -1), np.where(X_test_pred4 == -1)
# print("X_train_outlier_index4 outlier: ", X_train_outlier_index4)
# outlier_index5 = np.where(y_pred5 == -1)
X_train_outlier_index6, X_test_outlier_index6 = np.where(
    X_train_pred6 == -1), np.where(X_test_pred6 == -1)

novelty_X_train_outlier_index2 = np.where(
    novelty_X_train_pred2 == -1)
# print("novelty_X_train_outlier_index2 aaaaaaa: ", novelty_X_train_outlier_index2)

# temp2train = list((int(j) for i in X_train_outlier_index2 for j in i))
# temp2test = list((int(j) for i in X_test_outlier_index2 for j in i))
# print("X_train_outlier_index2 outlier: ", len(temp2train))
# print("X_test_outlier_index2 outlier: ", len(temp2test))
# # print("outlier_index2 inlier: ", np.where(
# #     X_train_pred2 == -1))
# temp3train = list((int(j) for i in X_train_outlier_index3 for j in i))
# temp3test = list((int(j) for i in X_test_outlier_index3 for j in i))
# print("X_train_outlier_index3 outlier: ", len(temp3train))
# print("X_test_outlier_index3 outlier: ", len(temp3test))
# # print("outlier_index3 inlier: ", np.where(
# #     X_train_pred3 == -1))
# temp4train = list((int(j) for i in X_train_outlier_index4 for j in i))
# temp4test = list((int(j) for i in X_test_outlier_index4 for j in i))
# print("X_train_outlier_index4 outlier: ", len(temp4train))
# print("X_test_outlier_index4 outlier: ", len(temp4test))
# # print("outlier_index4 inlier: ", np.where(
# #     X_train_pred4 == -1))
# temp6train = list((int(j) for i in X_train_outlier_index6 for j in i))
# temp6test = list((int(j) for i in X_test_outlier_index6 for j in i))
# print("X_train_outlier_index6 outlier: ", len(temp6train))
# print("X_test_outlier_index6 outlier: ", len(temp6test))
# print("outlier_index6 inlier: ", np.where(
# X_train_pred6 == -1))
# print("X_train_outlier_index6 outlier: ", X_train_outlier_index6)
# outlier_index7 = np.where(y_pred7 == -1)

# filter outlier values
X_train_outlier_values2, X_test_outlier_values2 = df.iloc[
    X_train_outlier_index2], df.iloc[X_test_outlier_index2]
# print("X_train_outlier_values2 outlier: ", X_train_outlier_values2)
# selection = pd.DataFrame(data=outlier_index2)
# print(selection)
X_train_outlier_values3, X_test_outlier_values3 = df.iloc[
    X_train_outlier_index3], df.iloc[X_test_outlier_index3]
# print("X_train_outlier_values3 outlier: ", len(X_train_outlier_values3))
X_train_outlier_values4, X_test_outlier_values4 = df.iloc[
    X_train_outlier_index4], df.iloc[X_test_outlier_index4]
# print("X_train_outlier_values4 outlier: ", len(X_train_outlier_values4))
# outlier_values5 = df.iloc[outlier_index5]
X_train_outlier_values6, X_test_outlier_values6 = df.iloc[
    X_train_outlier_index6], df.iloc[X_test_outlier_index6]
# print("X_train_outlier_values6 outlier: ", len(X_train_outlier_values6))
# outlier_values7 = df.iloc[outlier_index7]
novelty_X_train_outlier_values2 = df.iloc[novelty_X_train_outlier_index2]

# print("novelty_X_train_outlier_values2 aaaaaaa: ",
#       novelty_X_train_outlier_values2)


# # # print("New, normal observation accuracy:", list(
# # #     y_pred_test).count(1)/y_pred_test.shape[0])
# # # print("IF Outliers accuracy:", list(
# # #     y_pred2).count(-1)/y_pred2.shape[0])
# # # print("OCSVM Outliers accuracy:", list(
# # #     y_pred3).count(-1)/y_pred3.shape[0])
# # # print("LOF Outliers accuracy:", list(
# # #     y_pred4).count(-1)/y_pred4.shape[0])
# # # # print("Outliers accuracy:", list(
# # # #     y_pred5).count(-1)/y_pred5.shape[0])
# # # print("RC Outliers accuracy:", list(
# # #     y_pred6).count(-1)/y_pred6.shape[0])
# # # print("Outliers accuracy:", list(
# # #     y_pred7).count(-1)/y_pred7.shape[0])

# # # print("baseline: ", baseline.index.values)
# # # print("baseline type: ", type(baseline.index.values))
# # # print("outlier_values6: ", outlier_values6.index.values)
# # # print("outlier_values6 type: ", type(outlier_values6.index.values))

# # count percentage
# X_train_baseline1, X_train_reconfiguration1, X_train_perturbation1, X_train_mixed1 = np.isin(
#     X_train_outlier_values2.index.values, baseline_X_train.index.values), np.isin(
#     X_train_outlier_values2.index.values, reconfiguration_X_train.index.values), np.isin(
#     X_train_outlier_values2.index.values, perturbation_X_train.index.values), np.isin(
#     X_train_outlier_values2.index.values, mixed_X_train.index.values)
# # print("X_train_baseline1: ", X_train_baseline1)
# # print("X_train_reconfiguration1: ", X_train_reconfiguration1)
# # print("X_train_perturbation1: ", X_train_perturbation1)
# # print("X_train_mixed1: ", type(X_train_mixed1))
# X_test_baseline1, X_test_reconfiguration1, X_test_perturbation1, X_test_mixed1 = np.isin(
#     X_test_outlier_values2.index.values, baseline_X_test.index.values), np.isin(
#     X_test_outlier_values2.index.values, reconfiguration_X_test.index.values), np.isin(
#     X_test_outlier_values2.index.values, perturbation_X_test.index.values), np.isin(
#     X_test_outlier_values2.index.values, mixed_X_test.index.values)
# # X_train_alg1 = np.count_nonzero(X_train_baseline1 == True)/len(baseline_X_train), np.count_nonzero(X_train_reconfiguration1 == True) / \
# #     len(reconfiguration_X_train), np.count_nonzero(
# #         X_train_perturbation1 == True)/len(perturbation_X_train), np.count_nonzero(X_train_mixed1 == True)/len(mixed_X_train)
# # X_test_alg1 = np.count_nonzero(X_test_baseline1 == True)/X_test_baseline1.size, np.count_nonzero(X_test_reconfiguration1 == True) / \
# #     X_test_reconfiguration1.size, np.count_nonzero(
# #     X_test_perturbation1 == True)/X_test_perturbation1.size, np.count_nonzero(X_test_mixed1 == True)/X_test_mixed1.size
# print("============ ISOLATION FOREST ============")
# print("Number of outlier in training baseline: ",
#       np.count_nonzero(X_train_baseline1 == True))
# print("Number of outlier in training reconfiguration: ",
#       np.count_nonzero(X_train_reconfiguration1 == True))
# print("Number of outlier in training perturbation: ",
#       np.count_nonzero(X_train_perturbation1 == True))
# print("Number of outlier in training mixed: ",
#       np.count_nonzero(X_train_mixed1 == True))
# print("Number of outlier in testing baseline: ",
#       np.count_nonzero(X_test_baseline1 == True))
# print("Number of outlier in testing reconfiguration: ",
#       np.count_nonzero(X_test_reconfiguration1 == True))
# print("Number of outlier in testing perturbation: ",
#       np.count_nonzero(X_test_perturbation1 == True))
# print("Number of outlier in testing mixed: ",
#       np.count_nonzero(X_test_mixed1 == True))


# # print("Outlier percentage of X_train_alg1: ", X_train_alg1)
# # print("Outlier percentage of X_test_alg1: ", X_test_alg1)

# X_train_baseline2, X_train_reconfiguration2, X_train_perturbation2, X_train_mixed2 = np.isin(
#     X_train_outlier_values3.index.values, baseline_X_train.index.values), np.isin(
#     X_train_outlier_values3.index.values, reconfiguration_X_train.index.values), np.isin(
#     X_train_outlier_values3.index.values, perturbation_X_train.index.values), np.isin(
#     X_train_outlier_values3.index.values, mixed_X_train.index.values)
# X_test_baseline2, X_test_reconfiguration2, X_test_perturbation2, X_test_mixed2 = np.isin(
#     X_test_outlier_values3.index.values, baseline_X_test.index.values), np.isin(
#     X_test_outlier_values3.index.values, reconfiguration_X_test.index.values), np.isin(
#     X_test_outlier_values3.index.values, perturbation_X_test.index.values), np.isin(
#     X_test_outlier_values3.index.values, mixed_X_test.index.values)
# # X_train_alg2 = np.count_nonzero(X_train_baseline2 == True)/X_train_baseline2.size, np.count_nonzero(X_train_reconfiguration2 == True) / \
# #     X_train_reconfiguration2.size, np.count_nonzero(
# #         X_train_perturbation2 == True)/X_train_perturbation2.size, np.count_nonzero(X_train_mixed2 == True)/X_train_mixed2.size
# # X_test_alg2 = np.count_nonzero(X_test_baseline2 == True)/X_test_baseline2.size, np.count_nonzero(X_test_reconfiguration2 == True) / \
# #     X_test_reconfiguration2.size, np.count_nonzero(
# #     X_test_perturbation2 == True)/X_test_perturbation2.size, np.count_nonzero(X_test_mixed2 == True)/X_test_mixed2.size
# # print("Outlier percentage of X_train_alg2: ", X_train_alg2)
# # print("Outlier percentage of X_test_alg2: ", X_test_alg2)
# print("============ ONE CLASS SVM ============")
# print("Number of outlier in training baseline: ",
#       np.count_nonzero(X_train_baseline2 == True))
# print("Number of outlier in training reconfiguration: ",
#       np.count_nonzero(X_train_reconfiguration2 == True))
# print("Number of outlier in training perturbation: ",
#       np.count_nonzero(X_train_perturbation2 == True))
# print("Number of outlier in training mixed: ",
#       np.count_nonzero(X_train_mixed2 == True))
# print("Number of outlier in testing baseline: ",
#       np.count_nonzero(X_test_baseline2 == True))
# print("Number of outlier in testing reconfiguration: ",
#       np.count_nonzero(X_test_reconfiguration2 == True))
# print("Number of outlier in testing perturbation: ",
#       np.count_nonzero(X_test_perturbation2 == True))
# print("Number of outlier in testing mixed: ",
#       np.count_nonzero(X_test_mixed2 == True))

# X_train_baseline3, X_train_reconfiguration3, X_train_perturbation3, X_train_mixed3 = np.isin(
#     X_train_outlier_values4.index.values, baseline_X_train.index.values), np.isin(
#     X_train_outlier_values4.index.values, reconfiguration_X_train.index.values), np.isin(
#     X_train_outlier_values4.index.values, perturbation_X_train.index.values), np.isin(
#     X_train_outlier_values4.index.values, mixed_X_train.index.values)
# X_test_baseline3, X_test_reconfiguration3, X_test_perturbation3, X_test_mixed3 = np.isin(
#     X_test_outlier_values4.index.values, baseline_X_test.index.values), np.isin(
#     X_test_outlier_values4.index.values, reconfiguration_X_test.index.values), np.isin(
#     X_test_outlier_values4.index.values, perturbation_X_test.index.values), np.isin(
#     X_test_outlier_values4.index.values, mixed_X_test.index.values)
# # X_train_alg3 = np.count_nonzero(X_train_baseline3 == True)/X_train_baseline3.size, np.count_nonzero(X_train_reconfiguration3 == True) / \
# #     X_train_reconfiguration3.size, np.count_nonzero(
# #         X_train_perturbation3 == True)/X_train_perturbation3.size, np.count_nonzero(X_train_mixed3 == True)/X_train_mixed3.size
# # X_test_alg3 = np.count_nonzero(X_test_baseline3 == True)/X_test_baseline3.size, np.count_nonzero(X_test_reconfiguration3 == True) / \
# #     X_test_reconfiguration3.size, np.count_nonzero(
# #     X_test_perturbation3 == True)/X_test_perturbation3.size, np.count_nonzero(X_test_mixed3 == True)/X_test_mixed3.size
# # print("Outlier percentage of X_train_alg3: ", X_train_alg3)
# # print("Outlier percentage of X_test_alg3: ", X_test_alg3)
# print("============ LOCAL OUTLIER FACTOR ============")
# print("Number of outlier in training baseline: ",
#       np.count_nonzero(X_train_baseline3 == True))
# print("Number of outlier in training reconfiguration: ",
#       np.count_nonzero(X_train_reconfiguration3 == True))
# print("Number of outlier in training perturbation: ",
#       np.count_nonzero(X_train_perturbation3 == True))
# print("Number of outlier in training mixed: ",
#       np.count_nonzero(X_train_mixed3 == True))
# print("Number of outlier in testing baseline: ",
#       np.count_nonzero(X_test_baseline3 == True))
# print("Number of outlier in testing reconfiguration: ",
#       np.count_nonzero(X_test_reconfiguration3 == True))
# print("Number of outlier in testing perturbation: ",
#       np.count_nonzero(X_test_perturbation3 == True))
# print("Number of outlier in testing mixed: ",
#       np.count_nonzero(X_test_mixed3 == True))

# X_train_baseline4, X_train_reconfiguration4, X_train_perturbation4, X_train_mixed4 = np.isin(
#     X_train_outlier_values6.index.values, baseline_X_train.index.values), np.isin(
#     X_train_outlier_values6.index.values, reconfiguration_X_train.index.values), np.isin(
#     X_train_outlier_values6.index.values, perturbation_X_train.index.values), np.isin(
#     X_train_outlier_values6.index.values, mixed_X_train.index.values)
# X_test_baseline4, X_test_reconfiguration4, X_test_perturbation4, X_test_mixed4 = np.isin(
#     X_test_outlier_values6.index.values, baseline_X_test.index.values), np.isin(
#     X_test_outlier_values6.index.values, reconfiguration_X_test.index.values), np.isin(
#     X_test_outlier_values6.index.values, perturbation_X_test.index.values), np.isin(
#     X_test_outlier_values6.index.values, mixed_X_test.index.values)
# # X_train_alg4 = np.count_nonzero(X_train_baseline4 == True)/X_train_baseline4.size, np.count_nonzero(X_train_reconfiguration4 == True) / \
# #     X_train_reconfiguration4.size, np.count_nonzero(
# #         X_train_perturbation4 == True)/X_train_perturbation4.size, np.count_nonzero(X_train_mixed4 == True)/X_train_mixed4.size
# # X_test_alg4 = np.count_nonzero(X_test_baseline4 == True)/X_test_baseline4.size, np.count_nonzero(X_test_reconfiguration4 == True) / \
# #     X_test_reconfiguration4.size, np.count_nonzero(
# #     X_test_perturbation4 == True)/X_test_perturbation4.size, np.count_nonzero(X_test_mixed4 == True)/X_test_mixed4.size
# # print("Outlier percentage of X_train_alg4: ", X_train_alg4)
# # print("Outlier percentage of X_test_alg4: ", X_test_alg4)
# print("============ ROBUST COVARIANCE ============")
# print("Number of outlier in training baseline: ",
#       np.count_nonzero(X_train_baseline4 == True))
# print("Number of outlier in training reconfiguration: ",
#       np.count_nonzero(X_train_reconfiguration4 == True))
# print("Number of outlier in training perturbation: ",
#       np.count_nonzero(X_train_perturbation4 == True))
# print("Number of outlier in training mixed: ",
#       np.count_nonzero(X_train_mixed4 == True))
# print("Number of outlier in testing baseline: ",
#       np.count_nonzero(X_test_baseline4 == True))
# print("Number of outlier in testing reconfiguration: ",
#       np.count_nonzero(X_test_reconfiguration4 == True))
# print("Number of outlier in testing perturbation: ",
#       np.count_nonzero(X_test_perturbation4 == True))
# print("Number of outlier in testing mixed: ",
#       np.count_nonzero(X_test_mixed4 == True))


# # # print('Micro Precision: {:.2f}'.format(
# # #     precision_score(data["status"], y_pred6, average='micro')))
# # # print('Micro Recall: {:.2f}'.format(
# # #     recall_score(data["status"], y_pred6, average='micro')))
# # # print(
# # #     'Micro F1-score: {:.2f}\n'.format(f1_score(data["status"], y_pred6, average='micro')))

# # # print('Macro Precision: {:.2f}'.format(
# # #     precision_score(data["status"], y_pred6, average='macro')))
# # # print('Macro Recall: {:.2f}'.format(
# # #     recall_score(data["status"], y_pred6, average='macro')))
# # # print(
# # #     'Macro F1-score: {:.2f}\n'.format(f1_score(data["status"], y_pred6, average='macro')))

# # # print('Weighted Precision: {:.2f}'.format(
# # #     precision_score(data["status"], y_pred6, average='weighted')))
# # # print('Weighted Recall: {:.2f}'.format(
# # #     recall_score(data["status"], y_pred6, average='weighted')))
# # # print(
# # #     'Weighted F1-score: {:.2f}'.format(f1_score(data["status"], y_pred6, average='weighted')))
# # with np.printoptions(threshold=np.inf):
# #     print("data_X_train[status]: ", data_X_train["status"])
# #     print("X_train_pred3 : ", X_train_pred3)


# print('======= STAT REPORT IF =======')
# print(classification_report(data_X_train["status_ori"], data_X_train["pred2"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred2"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred2"]).ravel())
# print(classification_report(data_X_test["status_ori"], data_X_test["pred2"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred2"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred2"]).ravel())
# print(classification_report(
# data_X_test["status_ori"], data_X_test["pred2novelty"]))

# print('======= STAT REPORT OCSVM =======')
# print(classification_report(data_X_train["status"], data_X_train["pred3"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred3"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred3"]).ravel())
# print(classification_report(data_X_test["status"], data_X_test["pred3"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred3"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred3"]).ravel())

# print('======= STAT REPORT LOF =======')
# print(classification_report(data_X_train["status"], data_X_train["pred4"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred4"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred4"]).ravel())
# print(classification_report(data_X_test["status"], data_X_test["pred4"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred4"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred4"]).ravel())

# print('======= STAT REPORT RC =======')
# print(classification_report(data_X_train["status"], data_X_train["pred6"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred6"]))
# print(confusion_matrix(data_X_train["status"], data_X_train["pred6"]).ravel())
# print(classification_report(data_X_test["status"], data_X_test["pred6"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred6"]))
# print(confusion_matrix(data_X_test["status"], data_X_test["pred6"]).ravel())


# # tn_train, fp_train, fn_train, tp_train = confusion_matrix(
# #     data_X_train["status"], data_X_train["pred6"]).ravel()
# # cm_test = confusion_matrix(data_X_test["status"], X_test_pred4)
# # tn_test, fp_test, fn_test, tp_test = confusion_matrix(
# #     data_X_test["status"], X_test_pred4).ravel()
# # print("Confusion Matrix train: ")
# # print(cm_train)
# # print(confusion_matrix(data_X_train["status"], data_X_train["pred2"]))
# # print(tn_train, fp_train, fn_train, tp_train)
# # print("Confusion Matrix test: ")
# # print(cm_test)
# # print(tn_test, fp_test, fn_test, tp_test)

# # # # print('\nTraining accuracy: {:.2f}'.format(
# # # #     accuracy_score(X_train_pred4, data_X_train["status"])))
# # # # print('Testing accuracy: {:.2f}'.format(
# # # #     accuracy_score(X_test_pred4, data_X_test["status"])))

# # plot data
figsize = (12, 8)
fig1 = plt.figure(figsize=figsize)
plt.rcParams["font.family"] = "Times New Roman"
plt.subplots_adjust(left=None, bottom=None, right=None,
                    top=None, wspace=0.25, hspace=0.30)
plt.subplot(2, 2, 1)
training_inliers = plt.scatter(
    df_X_train["x"], df_X_train["y"], color="blue", label=r"Inliers in $D_{train}$", marker="+")
testing_inliers = plt.scatter(
    df_X_test["x"], df_X_test["y"], color="black", label=r"Inliers in $D_{test}$", marker='P')
training_outliers = plt.scatter(
    X_train_outlier_values2["x"], X_train_outlier_values2["y"], color="red", label=r"Outliers in $D_{train}$", marker='x')
training_outliers = plt.scatter(
    X_test_outlier_values2["x"], X_test_outlier_values2["y"], color="green", label=r"Outliers in $D_{test}$",  marker='X')
plt.title("Isolation Forest (IF)", fontsize=13)
plt.xlabel("1st Principal Component", fontsize=12)
plt.ylabel("2nd Principal Component", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=13)

plt.subplot(2, 2, 2)
training_inliers = plt.scatter(
    df_X_train["x"], df_X_train["y"], color="blue", label=r"Inliers in $D_{train}$", marker="+")
testing_inliers = plt.scatter(
    df_X_test["x"], df_X_test["y"], color="black", label=r"Inliers in $D_{test}$", marker='P')
training_outliers = plt.scatter(
    X_train_outlier_values3["x"], X_train_outlier_values3["y"], color="red", label=r"Outliers in $D_{train}$", marker='x')
training_outliers = plt.scatter(
    X_test_outlier_values3["x"], X_test_outlier_values3["y"], color="green", label=r"Outliers in $D_{test}$",  marker='X')
plt.title("One-class SVM (OCSVM)", fontsize=13)
plt.xlabel("1st Principal Component", fontsize=12)
plt.ylabel("2nd Principal Component", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=13)

plt.subplot(2, 2, 3)
training_inliers = plt.scatter(
    df_X_train["x"], df_X_train["y"], color="blue", label=r"Inliers in $D_{train}$", marker="+")
testing_inliers = plt.scatter(
    df_X_test["x"], df_X_test["y"], color="black", label=r"Inliers in $D_{test}$", marker='P')
training_outliers = plt.scatter(
    X_train_outlier_values4["x"], X_train_outlier_values4["y"], color="red", label=r"Outliers in $D_{train}$", marker='x')
training_outliers = plt.scatter(
    X_test_outlier_values4["x"], X_test_outlier_values4["y"], color="green", label=r"Outliers in $D_{test}$",  marker='X')
plt.title("Local Outlier Factor (LOF)", fontsize=13)
plt.xlabel("1st Principal Component", fontsize=12)
plt.ylabel("2nd Principal Component", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=13)

# # # plt.subplot(2, 3, 4)
# # # plt.title("DBSCAN", fontsize=15)
# # # plt.scatter(df["x"], df["y"], color="b", s=65, label="inliers")
# # # plt.scatter(outlier_values5["x"],
# # #             outlier_values5["y"], color="r", label="outliers")
# # # plt.xlabel("1st Principal Component", fontsize=12)
# # # plt.ylabel("2nd Principal Component", fontsize=12)
# # # plt.xticks(fontsize=11)
# # # plt.yticks(fontsize=11)
# # # plt.legend(fontsize=13)

plt.subplot(2, 2, 4)
training_inliers = plt.scatter(
    df_X_train["x"], df_X_train["y"], color="blue", label=r"Inliers in $D_{train}$", marker="+")
testing_inliers = plt.scatter(
    df_X_test["x"], df_X_test["y"], color="black", label=r"Inliers in $D_{test}$", marker='P')
training_outliers = plt.scatter(
    X_train_outlier_values6["x"], X_train_outlier_values6["y"], color="red", label=r"Outliers in $D_{train}$", marker='x')
training_outliers = plt.scatter(
    X_test_outlier_values6["x"], X_test_outlier_values6["y"], color="green", label=r"Outliers in $D_{test}$",  marker='X')
plt.title("Robust Covariance (RC)", fontsize=13)
plt.xlabel("1st Principal Component", fontsize=12)
plt.ylabel("2nd Principal Component", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=13)


# N = 4
# train_base = (0.003571428571, 0.1357142857, 0, 0)
# train_reconfig = (0.2271604938, 0.387654321,
#                   0.3283950617, 0.3308641975)
# train_perturb = (0.01025641026, 0.06153846154, 0, 0)
# train_mix = (0.08515283843, 0.3864628821,
#              0.002183406114, 0)
# test_base = (0.0119047619, 0.08333333333, 0, 0)
# test_reconfig = (0.2148148148, 0.5851851852, 0.3111111111, 0.3259259259)
# test_perturb = (0, 0.2307692308, 0, 0)
# test_mix = (0.09259259259, 0.5987654321, 0.01851851852, 0.006172839506)

# ind = np.arange(N)
# width = 0.18
# fig2 = plt.figure(figsize=(8, 6))

# plt.subplot(1, 2, 1)
# plt.bar(ind+0, train_base, width, label=r'Baseline ($E_{B}$)')
# plt.bar(ind+0.18, train_reconfig, width, label=r'Reconfiguration ($E_{R}$)')
# plt.bar(ind+0.36, train_perturb, width, label=r'Perturbation ($E_{P}$)')
# plt.bar(ind+0.54, train_mix, width,
#         label=r'Reconfiguration, Perturbation ($E_{RP}$)')
# plt.title(r"Outlier percentage in $D_{train}$ ")
# plt.ylabel('Outlier (%)')
# plt.xlabel('Algorithms')
# plt.ylim(0, 1)
# plt.xticks(ind + 0.3, ('IF', 'OCSVM', 'LOF', 'RC'))
# plt.legend(loc='best')

# plt.subplot(1, 2, 2)
# plt.bar(ind+0, test_base, width, label=r'Baseline ($E_{B}$)')
# plt.bar(ind+0.18, test_reconfig, width, label=r'Reconfiguration ($E_{R}$)')
# plt.bar(ind+0.36, test_perturb, width, label=r'Perturbation ($E_{P}$)')
# plt.bar(ind+0.54, test_mix, width,
#         label=r'Reconfiguration, Perturbation ($E_{RP}$)')
# plt.title(r"Outlier percentage in $D_{test}$ ")
# plt.ylabel('Outlier (%)')
# plt.xlabel('Algorithms')
# plt.ylim(0, 1)
# plt.xticks(ind + 0.3, ('IF', 'OCSVM', 'LOF', 'RC'))
# plt.legend(loc='best')


plt.show()
