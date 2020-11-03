import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

df = pd.read_csv("week2_data.csv" , comment='#')
print(df.head())
X1=df.iloc[:,0]
X2=df.iloc[:,1]
X=np.column_stack((X1, X2))
y = df.iloc[:,2]

#baseline model (Part C)
correct_predictions_baseline = 0
for i in range(0, len(y)):
    if 1 == y[i]:
        correct_predictions_baseline +=1

accuracy_baseline = correct_predictions_baseline/len(y)
print("accuracy baseline", accuracy_baseline)

#Part A
raw_data = {
    "x1": X1,
    "x2": X2,
    "class": y
}

df = pd.DataFrame(raw_data)
colours = np.where(raw_data["class"]==1,'r','b')
X1_transform = X1**2

transformed_data = {
    "x1": X1_transform,
    "x2": X2,
    "class": y
}

df2 = pd.DataFrame(transformed_data)
colours2 = np.where(transformed_data["class"]==1,'r','b')
X_transform = np.column_stack((X1**2, X2))

model = LogisticRegression()
model.fit(X_transform, y)
print("inital model", model.coef_, model.intercept_)

boundary_line = []
x_values = []
for i in range(-105, 105):
    i = i/100
    boundary_line.append(-(model.intercept_+ model.coef_[:,0]*(i**2))/ model.coef_[:,1])
    x_values.append(i)

y_pred = np.sign(model.intercept_+model.coef_[:,0]*X_transform[:,0] + model.coef_[:,1]*X_transform[:,1])
col = []
correct_predictions = 0
for i in range(0, len(y_pred)):
    if y_pred[i] == y[i]:
        correct_predictions +=1
    if y_pred[i] == -1:
        col.append('#FFFF00')    
    else:
        col.append('#00FFFF')

accuracy = correct_predictions/len(y_pred)
print("initial accuracy", accuracy)

#partB
model = LinearSVC()
model.__init__(C=0.001)
model.fit(X_transform, y)
print("model C=0.001", model.coef_, model.intercept_)
model1 = {"intercept": model.intercept_, "coef": model.coef_}

model.__init__(C=0.1)
model.fit(X_transform, y)
print("model C=0.1", model.coef_, model.intercept_)
model2 = {"intercept": model.intercept_, "coef": model.coef_}

model.__init__(C=10)
model.fit(X_transform, y)
print("model C=10", model.coef_, model.intercept_)
model3 = {"intercept": model.intercept_, "coef": model.coef_}

model.__init__(C=1000, max_iter=5000) # is giving a warning about not converging (doesnt even when given 10000 iterations)
model.fit(X_transform, y)
print("model C=1000", model.coef_, model.intercept_)
model4 = {"intercept": model.intercept_, "coef": model.coef_}

boundary_lineC1 = []
boundary_lineC2 = []
boundary_lineC3 = []
boundary_lineC4 = []

for i in range(-105, 105):
    i = i/100
    boundary_lineC1.append(-(model1["intercept"]+ model1["coef"][:,0]*(i**2))/ model1["coef"][:,1])
    boundary_lineC2.append(-(model2["intercept"]+ model2["coef"][:,0]*(i**2))/ model2["coef"][:,1])
    boundary_lineC3.append(-(model3["intercept"]+ model3["coef"][:,0]*(i**2))/ model3["coef"][:,1])
    boundary_lineC4.append(-(model4["intercept"]+ model4["coef"][:,0]*(i**2))/ model4["coef"][:,1])

y_pred1 = np.sign(model1["intercept"] + model1["coef"][:,0] * X_transform[:,0] + model1["coef"][:,1] * X_transform[:,1])
y_pred2 = np.sign(model2["intercept"] + model2["coef"][:,0] * X_transform[:,0] + model2["coef"][:,1] * X_transform[:,1])
y_pred3 = np.sign(model3["intercept"] + model3["coef"][:,0] * X_transform[:,0] + model3["coef"][:,1] * X_transform[:,1])
y_pred4 = np.sign(model4["intercept"] + model4["coef"][:,0] * X_transform[:,0] + model4["coef"][:,1] * X_transform[:,1])

col1 = []
col2 = []
col3 = []
col4 = []

correct_predictions1 = 0
correct_predictions2 = 0
correct_predictions3 = 0
correct_predictions4 = 0

for i in range(0, len(y_pred1)):

    if y_pred1[i] == y[i]:
        correct_predictions1 +=1
    if y_pred2[i] == y[i]:
        correct_predictions2 +=1
    if y_pred3[i] == y[i]:
        correct_predictions3 +=1
    if y_pred4[i] == y[i]:
        correct_predictions4 +=1

    if y_pred1[i] == -1:
        col1.append('#FFFF00')    
    else:
        col1.append('#00FFFF')

    if y_pred2[i] == -1:
        col2.append('#FFFF00')    
    else:
        col2.append('#00FFFF')

    if y_pred3[i] == -1:
        col3.append('#FFFF00')    
    else:
        col3.append('#00FFFF')

    if y_pred4[i] == -1:
        col4.append('#FFFF00')    
    else:
        col4.append('#00FFFF')

accuracy1 = correct_predictions1/len(y_pred)
accuracy2 = correct_predictions2/len(y_pred)
accuracy3 = correct_predictions3/len(y_pred)
accuracy4 = correct_predictions4/len(y_pred)

print("accuracy1", accuracy1)
print("accuracy2", accuracy2)
print("accuracy3", accuracy3)
print("accuracy4", accuracy4)

#Part C
X_transform2 = np.column_stack((X1, X2, X1**2, X2**2))

model = LogisticRegression()
model.fit(X_transform2, y)
model5 = {"intercept": model.intercept_, "coef": model.coef_}
print("model 5", model.coef_, model.intercept_)

y_pred5 = np.sign(model5["intercept"] + model5["coef"][:,0] * X_transform2[:,0] + model5["coef"][:,1] * X_transform2[:,1] + model5["coef"][:,2] * X_transform2[:,2] + model5["coef"][:,3] * X_transform2[:,3])

col5 = []
correct_predictions5 =0
for i in range(0, len(y_pred5)):
    if y_pred5[i] == y[i]:
        correct_predictions5 +=1

    if y_pred5[i] == -1:
        col5.append('#FFFF00')    
    else:
        col5.append('#00FFFF')

accuracy5 = correct_predictions5/len(y_pred5)
print("accuracy5", accuracy5)

boundary_line5 = []
for i in range(-105, 105):
    i = i/100
    a = model5["coef"][:,3]
    b = model5["coef"][:,1]
    c = (model5["intercept"] + (model5["coef"][:,0]*i) + (model5["coef"][:,2]*(i**2)))
    boundary_line5.append((-b + np.sqrt((b**2) -(4*a*c)))/2*a) #tried '-' and it gave bad predictions 


#graphs
red_patch = mpatches.Patch(color='red', label='Value=1')
blue_patch = mpatches.Patch(color='blue', label='Value=-1')
yellow_patch = mpatches.Patch(color='#00FFFF', label='Predicted=1')
cyan_patch = mpatches.Patch(color='#FFFF00', label='Predicted=-1')

df.plot.scatter(x="x1",y="x2",c=colours, title="raw data")
plt.legend(loc='upper left', handles=[red_patch, blue_patch])

df2.plot.scatter(x="x1",y="x2",c=colours2, title="transformed data")
plt.legend(loc='upper left', handles=[red_patch, blue_patch])
plt.xlabel("x1^2")

df.plot.scatter(x="x1",y="x2",c=colours, marker='o', s=40, title="Logistic Regression Model")
plt.scatter(X1, X2, c=col, marker='+', alpha=0.8, s=24)
plt.legend(loc='upper left', handles=[red_patch, blue_patch, yellow_patch, cyan_patch])
plt.plot(x_values, boundary_line, "--k")

df.plot.scatter(x="x1",y="x2",c=colours, marker='o', s=40, title="LinearSVM C=0.001")
plt.scatter(X1, X2, c=col1, marker='+', alpha=0.8, s=24)
plt.legend(loc='upper left', handles=[red_patch, blue_patch, yellow_patch, cyan_patch])
plt.plot(x_values, boundary_lineC1, "--k")

df.plot.scatter(x="x1",y="x2",c=colours, marker='o', s=40, title="LinearSVM C=0.1")
plt.scatter(X1, X2, c=col2, marker='+', alpha=0.8, s=24)
plt.legend(loc='upper left', handles=[red_patch, blue_patch, yellow_patch, cyan_patch])
plt.plot(x_values, boundary_lineC2, "--k")

df.plot.scatter(x="x1",y="x2",c=colours, marker='o', s=40, title="LinearSVM C=10")
plt.scatter(X1, X2, c=col3, marker='+', alpha=0.8, s=24)
plt.legend(loc='upper left', handles=[red_patch, blue_patch, yellow_patch, cyan_patch])
plt.plot(x_values, boundary_lineC3, "--k")

df.plot.scatter(x="x1",y="x2",c=colours, marker='o', s=40, title="LinearSVM C=1000")
plt.scatter(X1, X2, c=col4, marker='+', alpha=0.8, s=24)
plt.legend(loc='upper left', handles=[red_patch, blue_patch, yellow_patch, cyan_patch])
plt.plot(x_values, boundary_lineC4, "--k")

df.plot.scatter(x="x1",y="x2",c=colours, marker='o', s=40, title="Logistic Regression with extra features")
plt.scatter(X1, X2, c=col5, marker='+', alpha=0.8, s=24)
plt.legend(loc='upper left', handles=[red_patch, blue_patch, yellow_patch, cyan_patch])
plt.plot(x_values, boundary_line5, "--k")


Accuracy_array_for_C_values = [accuracy1, accuracy2, accuracy3, accuracy4]
C_values = [0.001, 0.1, 10, 1000]
plt.figure(9)
plt.plot(C_values, Accuracy_array_for_C_values,'--b')
plt.xscale("log")
plt.xlabel("C value")
plt.ylabel("accuracy")
plt.title("Accuracy vs C value for linear SVM")

plt.show()