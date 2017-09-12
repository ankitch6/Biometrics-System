from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt
#import requests



class Neural_Net(object):
    def __init__(self):
        self.iterations=20
        self.instances=232
        self.input_layer_size=62500;
        self.output_layer_size=36;
        self.hidden_layer_size=150;
        self.alpha=0.001
        self.w1=np.random.randn(self.input_layer_size,self.hidden_layer_size)
        self.w2=np.random.randn(self.hidden_layer_size,self.output_layer_size)
        self.bias1=np.random.randn(1,self.hidden_layer_size)
        self.bias2=np.random.randn(1,self.output_layer_size)
        
        
    def forward(self,X):
        self.z1=np.dot(X,self.w1)
        self.z1+=self.bias1;
        self.a1=self.sigmoid(self.z1)
        self.z2=np.dot(self.a1,self.w2)
        self.z2+=self.bias2;
        y_predicted=self.softmax_func(self.z2)
        return y_predicted
    
    def sigmoid(self,z):
           np.clip(z,-500,500)
           return 1/(1+np.exp(-z))
       
    def sigmoid_prime(self,z):
       
        return self.sigmoid(z)*(1-self.sigmoid(z));
    
    
    def cost_func_prime(self,X,y):
        cost_plot=np.zerose([self.instances,self.iterations])
        for cnt in range(0,self.iterations):
         print("iteration"+ "  "+str(cnt))
         for i in range(0,self.instances): 
          x1=np.zeros([1,self.input_layer_size])
          y1=np.zeros([1,self.output_layer_size])
          for j in range(0,self.input_layer_size):
              x1[0][j]=X[i][j]
          for j in range(0,self.output_layer_size):
              y1[0][j]=y[i][j]
          self.y_hat=self.forward(x1)
          cost1=self.cost_func(y1)
          cost_plot[i][cnt]=cost1
          
          delta3=self.y_hat-y1
          delta2=np.dot(delta3,self.w2.T)
          djdw2=np.dot(self.a1.T,delta3)
          djdw1=np.dot(x1.T,delta2)
          djdb2=delta3
          djdb1=delta2
          self.w2=self.w2-self.alpha*djdw2
          self.w1=self.w1-self.alpha*djdw1
          self.bias1=self.bias1-((self.alpha)*(djdb1))
          self.bias2=self.bias2-((self.alpha)*(djdb2))
        return cost_plot  
    
    
    
    def cost_func(self,y):
         y_tmp=((-1)*((y)*(np.log(self.y_hat))+((1-y)*(np.log(1-self.y_hat)))))
         return np.sum(y_tmp)/self.output_layer_size
        
        
    def softmax_func(self,zz):
        """Compute softmax values for each sets of scores in zz."""
        e_x = np.exp(zz - np.max(zz))
        return e_x / e_x.sum()
     
       
    def leaky_relu(self,z):
        m,n=z.shape
        for i in range(0,m):
            for j in range(0,n):
                if z[i][j]<0:
                    z[i][j]=0.01*z[i][j]
        return z
    def leaky_reluu(self,z):
        zz=np.multiply(0.01,z)
        return np.maximum(z,zz)
    def tanh(self,z):
        return (np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))



x_values=np.zeros([252,62500])
y_values=np.zeros([252,1])
base_name='\iris'
base_path='D:\iris_database'

#this is converting each image into grayscale values from a folder where the images are serially numbered

for i in range(1,37):
    for j in range(0,7):
        c=chr(ord('a')+j)
        full_path=base_path+base_name+str(i)+str(c)+'.bmp'
        img=Image.open(full_path).convert('L')
        new_img=img.resize((250,250))
        x_values[(i-1)*7+j]=np.array(new_img).reshape((1,62500))
        y_values[(i-1)*7+j][0]=i-1;

#this is converting the given output values into an output matrix that could be used for 
#neural net  i.e. m*1 type
y_values1=np.zeros([252,36])
for i in range(0,252):
    xxx=y_values[i][0]
    for j in range(0,36):
        if xxx==j:
            y_values1[i][j]=1                  
y_values=y_values1 

x_values1=x_values.T
#this is normalisation of data by taking mean and standard deviation

for i in range(0,62500):
    mean1=np.mean(x_values1[i])
    std1=np.std(x_values1[i])
    x_values1[i,:]-=mean1
    x_values1[i,:]/=std1
             
x_values=x_values1.T             


#the dataset has 252 images and we are randomly selecting 232 for training and 20 for testing

num=random.sample(range(252),20);
vis=np.zeros([252,1]);
for i in range(0,20):
   vis[num[i]]=1;## SEE HERE

     
x_train=np.zeros([232,62500]);
x_test=np.zeros([20,62500]);
y_test=np.zeros([20,36]);
y_train=np.zeros([232,36])
itrain=0;
itest=0;         

#dividing my dataset into training and testing datasets and corresponding output values also


for i in range(0,252):
    if(vis[i]==1):
        x_test[itest] =x_values[i];
        y_test[itest]=y_values[i];
        itest+=1;
    else:
        x_train[itrain]=x_values[i];
        y_train[itrain]=y_values[i];
        itrain+=1;
#creating object of the class
obj=Neural_Net()


#cost_instances will store the costfunction values for each iteration for every instance

cost_instances=obj.cost_func_prime(x_train,y_train)        
         
#using against to plot the cost_instances values
        
against=np.arange(obj.iterations);
                 
                 
for i in range(0,232):# it should be 125 here
    plt.plot(against,cost_instances[i])

plt.show()     


#counting total postives and hence predicting accuracy
tp=0;
for i in range(0,20):
  xx=x_test[i];
  xx=np.array([xx])
  yy=y_test[i]
  yy=np.array([yy])
  ypredict=obj.forward(xx)
  ypredict11=ypredict.argmax(axis=1)
  if(yy[0][ypredict11]==1):
   tp+=1


print("tp here is ",tp)
acc=tp/20
print("accuracy is:",acc) 
