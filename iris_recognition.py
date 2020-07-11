from PIL import Image
import numpy as np
import random
import os
import matplotlib.pyplot as plt
#import requests


np.seterr(all = 'ignore')
class Neural_Net(object):
    def __init__(self):
        self.epochs=20
        self.instances=140
        self.input_layer_size=10800;
        self.output_layer_size=20;
        self.hidden_layer1_size=530;
        self.hidden_layer2_size=440;
        self.alpha=0.001
        self.w1=np.random.randn(self.input_layer_size,self.hidden_layer1_size)
        self.w2=np.random.randn(self.hidden_layer1_size,self.hidden_layer2_size)
        self.w3=np.random.randn(self.hidden_layer2_size,self.output_layer_size)
        self.bias1=np.random.randn(1,self.hidden_layer1_size)
        self.bias2=np.random.randn(1,self.hidden_layer2_size)
        self.bias3=np.random.randn(1,self.output_layer_size)
        
        
    def forward(self,X):
        self.z1=np.dot(X,self.w1)+self.bias1
        self.a1=self.sigmoid(self.z1)
        self.z2=np.dot(self.a1,self.w2)+self.bias2
        self.a2=self.sigmoid(self.z2)
        self.z3=np.dot(self.a2,self.w3)+self.bias3
        #y_predicted=self.softmax_func(self.z3)
        y_predicted=self.sigmoid(self.z3)
        return y_predicted
    
    def sigmoid(self,z):
           #np.clip(z,-500,500)
           return 1/(1+np.exp(-z))
       
    def sigmoid_prime(self,z):   
        return self.sigmoid(z)*(1-self.sigmoid(z));
    
    
    def train(self,X,y):
        cost_plot=np.zeros([self.epochs])
        for epoch in range(0,self.epochs):
        #for ep in range(0,1):
            
            print("epoch: "+ "  "+str(epoch))
            mse=0.0
            for i in range(0,self.instances): 
                x1=np.zeros([1,self.input_layer_size])
                y1=np.zeros([1,self.output_layer_size])
                for j in range(0,self.input_layer_size):
                    x1[0][j]=X[i][j]
                for j in range(0,self.output_layer_size):
                    y1[0][j]=y[i][j]
                self.y_hat=self.forward(x1)
                cost=self.cost_func(y1)
                mse=mse+cost
                
                delta4=-(y1-self.y_hat)
                delta3=np.dot(delta4,self.w3.T)*self.sigmoid_prime(self.a2)
                delta2=np.dot(delta3,self.w2.T)*self.sigmoid_prime(self.a1)
                dw3=np.dot(self.a2.T,delta4)
                dw2=np.dot(self.a1.T,delta3)
                dw1=np.dot(x1.T,delta2)
                db3=delta4
                db2=delta3
                db1=delta2
                self.w3=self.w3-self.alpha*dw3
                self.w2=self.w2-self.alpha*dw2
                self.w1=self.w1-self.alpha*dw1
                self.bias1=self.bias1-((self.alpha)*(db1))
                self.bias2=self.bias2-((self.alpha)*(db2))
                self.bias3=self.bias3-((self.alpha)*(db3))
            #print "error="+str(mse)
            print mse
            cost_plot[epoch]=mse
            if mse<=0.01:
                break
        return cost_plot  
    
    def cost_func(self,y):
         #y_tmp=((-1)*((y)*(np.log(self.y_hat))+((1-y)*(np.log(1-self.y_hat)))))
         y_tmp=(y-self.y_hat)**2
         #return np.sum(y_tmp)/self.output_layer_size
         return np.sum(y_tmp)/(2*self.output_layer_size) 
        
    def softmax_func(self,zz):
        """Compute softmax values for each sets of scores in zz."""
        e_x = np.exp(zz - np.max(zz))
        return e_x / e_x.sum()




x_values=np.zeros([200,10800])
y_values=np.zeros([200,1])
base_path="input/"

#this is converting each image into grayscale values from a folder where the images are serially numbered

for i in range(1,21):
    eyeList=os.listdir(base_path+str(i)+"/")
    for j in range(0,10):
        #c=chr(ord('a')+j)
        full_path=base_path+str(i)+"/"+str(eyeList[j])
        img=Image.open(full_path).convert('L')
        new_img=img.resize((60,180))
        x_values[(i-1)*10+j]=np.array(new_img).reshape((1,10800))
        y_values[(i-1)*10+j][0]=i-1;
#print x_values[0]
#this is converting the given output values into an output matrix that could be used for 
#neural net  i.e. m*1 type
y_values1=np.zeros([200,20])
for i in range(0,200):
    xxx=y_values[i][0]
    for j in range(0,20):
        if xxx==j:
            y_values1[i][j]=1                  
y_values=y_values1 

x_values1=x_values.T
#this is normalisation of data by taking mean and standard deviation
mean1=np.zeros([1,10800])
std1=np.zeros([1,10800])
for i in range(0,10800):
    mean1[0][i]=np.mean(x_values1[i])
    std1[0][i]=np.std(x_values1[i])
    x_values1[i,:]-=mean1[0][i]
    x_values1[i,:]/=std1[0][i]
             
x_values=x_values1.T             

'''
#the dataset has 200 images and we are randomly selecting 180 for training and 20 for testing

num=random.sample(range(200),20);
vis=np.zeros([200,1]);
for i in range(0,20):
   vis[num[i]]=1;## SEE HERE
'''
     
x_train=np.zeros([140,10800]);
x_test=np.zeros([60,10800]);
y_test=np.zeros([60,20]);
y_train=np.zeros([140,20])
itrain=0;
itest=0;         

#dividing my dataset into training and testing datasets and corresponding output values also
'''

for i in range(0,200):
    if(vis[i]==1):
        x_test[itest] =x_values[i];
        y_test[itest]=y_values[i];
        itest+=1;
    else:
        x_train[itrain]=x_values[i];
        y_train[itrain]=y_values[i];
        itrain+=1;
'''
for i in range(0,20):
    for j in range(0,7):
        x_train[itrain]=x_values[10*i+j];
        y_train[itrain]=y_values[10*i+j];
        itrain+=1;
    for j in range(7,10):
        x_test[itest] =x_values[10*i+j];
        y_test[itest]=y_values[10*i+j];
        itest+=1;
        
#creating object of the class
obj=Neural_Net()
#cost_values will store the costfunction values for each iteration

cost_values=obj.train(x_train,y_train)        

#using against to plot the cost_values values
        
iterat=np.arange(obj.epochs)*obj.instances;
plt.plot(iterat,cost_values)

plt.show()  
 

'''
#counting total postives and hence predicting accuracy
tp=0;
for i in range(0,200):
    xx=x_values[i];
    xx=np.array([xx])
    yy=y_values[i]
    yy=np.array([yy])
    ypredict=obj.forward(xx)
    ypredict11=ypredict.argmax(axis=1)
    if(yy[0][ypredict11]==1):
        tp+=1
    else:
        print "S"+str(1001+int(i/10))+"L"+str(1+int(i%10))
        
print("tp here is ",tp)
acc=tp/200.0
print("accuracy is:",acc)
'''

#counting total postives and hence predicting accuracy
tp=0;
for i in range(0,60):
    xx=x_test[i];
    xx=np.array([xx])
    yy=y_test[i]
    yy=np.array([yy])
    ypredict=obj.forward(xx)
    ypredict11=ypredict.argmax(axis=1)
    if(yy[0][ypredict11]==1):
        tp+=1

print("tp here is ",tp)
acc=tp/60.0
print("accuracy is:",acc) 


xinput=np.zeros([1,10800])
iter=0
while iter<3:
    ip=raw_input("Enter image name: ")
    full_path="temp/"+str(ip)
    img=Image.open(full_path).convert('L')
    new_img=img.resize((60,180))
    xinput[0]=np.array(new_img).reshape((1,10800))

    #this is normalisation of data by taking mean and standard deviation
    xinput1=xinput.T
    for i in range(0,10800):
        xinput1[i,:]-=mean1[0][i]
        xinput1[i,:]/=std1[0][i]             
    xinput=xinput1.T 
    #END normalization

    xt=xinput[0]
    xt=np.array([xt])
    yp=obj.forward(xt)
    #print yp
    yp11=yp.argmax(axis=1)
    print "Recognized as: Person #"+str( yp11[0]+1)
    iter=iter+1