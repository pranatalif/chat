from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycm import *
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor

from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# chaos = pd.read_csv(
#     '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/Second Paper/Results/Scenario 2/net/TestSDN Data Set S2 B - Mixed 2.csv')
chaos = pd.read_csv(
    '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/Second Paper/Results/Scenario 2/net/error-detection.csv')
# chaos = pd.read_csv(
#     '/Users/aalif/Documents/Google Drive/4.Prancis/Paper/Second Paper/Results/Scenario 2/net/TestSDN Data Set S2 B - 35-feature.csv')
chaos.head()


# print(chaos.head(3))
# print(chaos.shape)


baseline = len(chaos[chaos.target == 1])
print("number of baseline samples: ", baseline)
print(chaos.columns)

feature_req = list(chaos.columns[1:3])
feature_code = list(chaos.columns[3:7])
feature_nginx = list(chaos.columns[7:1])
feature_all = list(chaos.columns[0:42])


X_train, X_test, Y_train, Y_test = train_test_split(chaos, chaos.target, test_size=0.25,
                                                    stratify=chaos.target, random_state=30)



# Now Apply PCA
scaler1 = StandardScaler()
scaler1.fit(chaos)
feature_scaled = scaler1.transform(chaos)

pca1 = PCA(n_components=4)
pca1.fit(feature_scaled)
feature_scaled_pca = pca1.transform(feature_scaled)
print("shape of the scaled and 'PCA'ed features: ", np.shape(feature_scaled_pca))
res = pd.DataFrame(pca1.transform(feature_scaled))

feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))

print("Variance Ratio of the 4 Principal Components Ananlysis: ", feat_var_rat)


chaos_target_list = chaos.target.tolist()
# print(type(chaos_target_list))
# print (cancer_target_list)
# print (type(yl))
feature_scaled_pca_X0 = feature_scaled_pca[:, 0]
feature_scaled_pca_X1 = feature_scaled_pca[:, 1]
feature_scaled_pca_X2 = feature_scaled_pca[:, 2]
feature_scaled_pca_X3 = feature_scaled_pca[:, 3]
# print("feature_scaled_pca_X0: ", feature_scaled_pca_X0)
# print("type: ", type(feature_scaled_pca_X0))
# print("feature_scaled_pca_X1: ", feature_scaled_pca_X1)
# print("size: ", feature_scaled_pca_X1.shape)

# pca_arr = np.array(feature_scaled_pca_X0)
# # pca_arr.append(feature_scaled_pca_X0, axis=0)
# pca_arr = np.append(feature_scaled_pca_X1.values)
# pca_arr = np.array(2,)
# pca_arr = np.concatenate(
#     (feature_scaled_pca[:, 0], feature_scaled_pca[:, 1]), axis=1)
# pca_arr = feature_scaled_pca_X0.add(feature_scaled_pca_X1)
# print("pca_array : ", pca_arr)
# aaa = np.concatenate(
#     (feature_scaled_pca_X0, feature_scaled_pca_X1))
pca_array = numpy.asarray([feature_scaled_pca_X0, feature_scaled_pca_X1])
# print("gabungan: ", type(pca_array))
numpy.savetxt("pca_array1.csv", pca_array.transpose(), delimiter=",")

labels = chaos_target_list
# print(labels)
colordict = {1: 'lime', 2: 'purple', 3: 'red',
             4: 'blue', 5: 'green'}
#  E$_B$
# piclabel = {1: 'Baseline (E$_B$)', 2: 'Recofiguration (E$_R$)',
#             3: 'Perturbation (E$_P$)', 4: r'Reconfiguration, Perturbation (E$_{RP}$)', 5: 'Perturbation, Perturbation'}
piclabel = {1: 'E$_B$', 2: 'E$_R$',
            3: 'E$_P$', 4: 'E$_{RP}$'}
markers = {1: 'o', 2: '*', 3: 'd', 4: '+', 5: '^'}
alphas = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6, 5: 0.7}

# Z = np.array(res)
# figsize = (12, 7)
# plt.figure(figsize=figsize)
# plt.title("IsolationForest")
# plt.contourf(Z, cmap=plt.cm.Blues_r)

# b1 = plt.scatter(res[0], res[1], c='blue',
#                  s=40, label="normal points")

# b1 = plt.scatter(res.iloc[outlier_index, 0], res.iloc[outlier_index, 1], c='red',
#                  s=40,  edgecolor="red", label="predicted outliers")
# plt.legend(loc="upper right")


fig = plt.figure(figsize=(8.5, 7))
plt.rcParams["font.family"] = "Times New Roman"
# plt.subplot(1, 2, 1)
for l in np.unique(labels):
    ix = np.where(labels == l)
    plt.scatter(feature_scaled_pca_X0[ix], feature_scaled_pca_X1[ix], c=colordict[l],
                label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])
    # print("feature_scaled_pca_X0[ix]: ", feature_scaled_pca_X0[ix])
    # print("type: ", type(feature_scaled_pca_X0[ix]))
    # print("feature_scaled_pca_X1[ix]: ", feature_scaled_pca_X1[ix])
    # print("size: ", feature_scaled_pca_X1[ix].shape)

plt.xlabel("1st Principal Component", fontsize=23)
plt.ylabel("2nd Principal Component", fontsize=23)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.legend(fontsize=24)


plt.show()

# plt.subplot(1, 2, 2)
# for l1 in np.unique(labels):
#     ix1 = np.where(labels == l1)
#     plt.scatter(feature_scaled_pca_X2[ix1], feature_scaled_pca_X3[ix1], c=colordict[l1],
#                 label=piclabel[l1], s=40, marker=markers[l1], alpha=alphas[l1])
# plt.xlabel("Third Principal Component", fontsize=15)
# plt.ylabel("Fourth Principal Component", fontsize=15)
# plt.xticks(fontsize=13)
# plt.yticks(fontsize=13)

# plt.legend(fontsize=15)


# plt.savefig('Cancer_labels_PCAs.png', dpi=200)
# plt.show()

pipe_steps = [('scaler', StandardScaler()), ('pca', PCA()),
              ('SupVM', SVC(kernel='rbf'))]

check_params = {
    'pca__n_components': [4],
    'SupVM__C': [0.1, 0.5, 1, 10, 30, 40, 50, 75, 100, 500, 1000],
    'SupVM__gamma': [0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}


pipeline = Pipeline(pipe_steps)


print("Start Fitting Training Data")
for cv in tqdm(range(4, 6)):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
    create_grid.fit(X_train, Y_train)
    print("score for %d fold CV := %3.2f" %
          (cv, create_grid.score(X_test, Y_test)))
    print("!!!!!!!! Best-Fit Parameters From Training Data !!!!!!!!!!!!!!")
    print(create_grid.best_params_)

print("out of the loop")
print("grid best params: ", create_grid.best_params_)

# print(X_test)
# print("Ytest: ", Y_test)


Y_pred = create_grid.predict(X_test)
# print("Ypred: ", Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix: \n")
print(cm)


# ------

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(Y_test, Y_pred)))
# print("y_test shapeeee: ", Y_test)
# print("Y_pred shapeeee: ", Y_pred)
# print("y_test typeee: ", type(Y_test))
# print("Y_pred typeee: ", type(Y_pred))

print('Micro Precision: {:.2f}'.format(
    precision_score(Y_test, Y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(
    recall_score(Y_test, Y_pred, average='micro')))
print(
    'Micro F1-score: {:.2f}\n'.format(f1_score(Y_test, Y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(
    precision_score(Y_test, Y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(
    recall_score(Y_test, Y_pred, average='macro')))
print(
    'Macro F1-score: {:.2f}\n'.format(f1_score(Y_test, Y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(
    precision_score(Y_test, Y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(
    recall_score(Y_test, Y_pred, average='weighted')))
print(
    'Weighted F1-score: {:.2f}'.format(f1_score(Y_test, Y_pred, average='weighted')))

print('\nClassification Report\n')
print(classification_report(Y_test, Y_pred,
                            target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4']))

# y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
# y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
# cm2 = ConfusionMatrix(actual_vector=y_actu, predict_vector=y_pred)
# # cm2 = ConfusionMatrix(actual_vector=Y_test, predict_vector=Y_pred)
# print(cm2.classes)
# print(cm2)

# print("Y_actu: ", type(y_actu))
# print("Y_testt: ", type(Y_test))
# print("Y_predd: ", type(Y_pred))
# print("Y_testt coba ambil column: ", Y_test.iat[0])

# ------

df_cm = pd.DataFrame(cm, range(4), range(4))

sns.heatmap(df_cm, annot=True, cbar=False)
plt.title("Confusion Matrix", fontsize=14)
# plt.savefig("Result-pca.png", dpi=300)


scaler1 = StandardScaler()
scaler1.fit(X_test)
X_test_scaled = scaler1.transform(X_test)

# scaler2 = StandardScaler()
# scaler2.fit(X_train)
# X_train_scaled = scaler2.transform(X_train)

pca2 = PCA(n_components=2)
X_test_scaled_reduced = pca2.fit_transform(X_test_scaled)

# pca3 = PCA(n_components=2)
# X_train_scaled_reduced = pca3.fit_transform(X_train_scaled)


# svm_model = SVC(kernel='rbf', C=float(create_grid.best_params_['SupVM__C']),
#                 gamma=float(create_grid.best_params_['SupVM__gamma']))

svm_model = SVC(kernel='rbf', C=1., gamma=0.5)

classify = svm_model.fit(X_test_scaled_reduced, Y_test)
# classify2 = svm_model.fit(X_train_scaled_reduced, Y_train)


def plot_contours(ax, clf, xx, yy, **params):
    # print("ax: ", ax)
    # print("classify: ", classify)
    # print("xx: ", xx.shape)
    # print("yy: ", yy.ravel().shape)
    # print("np.c_[xx.ravel(), yy.ravel()]: ",
    #       np.c_[xx.ravel(), yy.ravel()].shape)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # print("Z", Z)
    # print("Z", Z.ravel().shape)
    print('initial decision function shape; ', np.shape(Z))
    # Z = np.rollaxis(Z, 1, 0)
    # print("Z after rollaxis", Z)
    # print("Z", Z.shape)
    Z = Z.reshape(xx.shape)
    print('after reshape: ', np.shape(Z))
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))  # ,
    # np.arange(z_min, z_max, h))
    return xx, yy


X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)


fig, ax = plt.subplots(figsize=(8.5, 7))
fig.patch.set_facecolor('white')

Y_tar_list = Y_test.tolist()
yl1 = [int(target1) for target1 in Y_tar_list]
labels1 = yl1

cdict1 = {1: 'lime', 2: 'purple', 3: 'red',
          4: 'blue', 5: 'green'}
# labl1 = {1: 'Baseline (E$_B$)', 2: 'Recofiguration (E$_R$)',
#          3: 'Perturbation (E$_P$)', 4: r'Reconfiguration, Perturbation (E$_{RP}$)', 5: 'Perturbation, Perturbation'}

labl1 = {1: 'E$_B$', 2: 'E$_R$',
         3: 'E$_P$', 4: 'E$_{RP}$'}
marker1 = {1: 'o', 2: '*', 3: 'd', 4: '+'}
alpha1 = {1: 0.3, 2: 0.4, 3: 0.5, 4: 0.6}

for l1 in np.unique(labels1):
    ix1 = np.where(labels1 == l1)
    ax.scatter(X0[ix1], X1[ix1], c=cdict1[l1], label=labl1[l1],
               s=70, marker=marker1[l1], alpha=alpha1[l1])

ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none',
           edgecolors='navy', label='Support Vectors')

plot_contours(ax, classify, xx, yy, cmap='seismic', alpha=0.4)
plt.xlabel("1st Principal Component", fontsize=23)
plt.ylabel("2nd Principal Component", fontsize=23)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.legend(fontsize=24)


plt.savefig('Result-svm.png', dpi=300)

print("ax: ", ax)
print("classify: ", classify)
print("xx: ", xx.shape)
print("yy: ", yy.shape)

plt.show()

