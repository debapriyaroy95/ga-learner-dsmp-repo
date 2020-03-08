# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here

data = pd.read_csv(path)

data_sample = data.sample(n=sample_size,random_state=0)
sample_std =data_sample['installment'].std()
sample_mean = data_sample['installment'].mean()


margin_of_error = z_critical * (sample_std / math.sqrt(sample_size))

confidence_interval = (sample_mean - margin_of_error ,sample_mean +  margin_of_error)

true_mean = data['installment'].mean()

print(true_mean)

print(confidence_interval)


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])
fig,axes =  plt.subplots(3,1)
for i in range (len(sample_size)):
    m=[]
    for j in range(1000):
        m.append(data['installment'].sample(n=sample_size[i]))
    mean_series= (pd.Series(m))
    #axes[i]=mean_series.value_counts().plot(kind='hist')

#plt.show()
print(mean_series)
#Code starts here



# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

data['int.rate'] = data['int.rate'].str.replace('%','')

print(data)
print(data.dtypes)
#data.astype({'int.rate': 'int32'}).dtypes
#data['int.rate'].apply(lambda x : x.astype('float'))

data['int.rate']=pd.to_numeric(data['int.rate'])
data['int.rate'] = (data['int.rate'])/100
#print(data)
z_statistic, p_value = ztest(data[data['purpose']=='small_business']['int.rate'],value=data['int.rate'].mean(),alternative='larger')

if p_value < 0.05 :
    inference = 'Reject'
else :
    inference = 'Accept'

print(inference)
#Code starts here



# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

z_statistic , p_value = ztest(x1 = data[data['paid.back.loan']=='No']['installment'] , x2 = data[data['paid.back.loan']=='Yes']['installment'])

if p_value < 0.05 :
    inference = 'Reject null Hypothesis'
else :
    inference = 'Accept null Hypothesis'

print(inference)

#Code starts here



# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

Yes = (data[data['paid.back.loan'] == 'Yes']['purpose']).value_counts()

No = (data[data['paid.back.loan'] == 'No']['purpose']).value_counts()

observed = pd.concat([Yes.transpose(),No.transpose()],axis=1 , keys= ['Yes','No'])
print(observed)

chi2, p, dof, ex = chi2_contingency(observed)

print(chi2)

if chi2 > critical_value :
    inference = 'Reject'
else :
    inference = 'Accept'

print(inference)
#Code starts here



