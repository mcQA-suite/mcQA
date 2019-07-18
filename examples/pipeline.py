from mcqa import MCQAData
from mcqa import Model

# create data
train_data_file = 'train.csv'
train_dataset = MCQAData().read(train_data_file)

# training
model = Model()
model.fit(train_dataset)
