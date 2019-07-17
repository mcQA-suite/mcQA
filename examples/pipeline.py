from mcqa import MCData
from mcqa import Model

# create data 
train_data_file = 'train.csv'
train_dataloader = MCData().read(train_data_file)

# training 
model = Model()
model.fit()