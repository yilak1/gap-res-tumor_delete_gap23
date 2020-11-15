import numpy
from dataset import TumorTrain,TumorTest
import matplotlib.pyplot as plt
from utils import  get_training_dataloader, get_test_dataloader
tumor_train_dataset = TumorTrain()
print(len(tumor_train_dataset))
tumor_training_loader = get_training_dataloader(
    num_workers=4,
    batch_size=30,
    shuffle=True
)
for i_batch, (sample_iamge, sample_label) in enumerate(tumor_training_loader):
    print(i_batch, sample_iamge.size(), sample_label.size())

# test
tumor_test_dataset = TumorTest()
print(len(tumor_test_dataset))
tumor_test_loader = get_test_dataloader(
    num_workers=4,
    batch_size=30,
    shuffle=True
)
for i_batch, (sample_iamge, sample_label) in enumerate(tumor_test_loader):
    print(i_batch, sample_iamge.size(), sample_label.size())




# for i in range(len(tumor_train_dataset)):
#     sample, label= tumor_train_dataset[i]
#     # print(sample.size(), label.size())