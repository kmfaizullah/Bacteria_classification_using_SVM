from Preprocess import Process
from evaluation import evaluation
from training import training

directory= 'data/'
test_size=0.001
random_state=109
validation_size=2
image_path='test/ecoli/4.png'
evaluation=evaluation()

train = training(directory,test_size,random_state)
model=train.train(validation_size)
score=evaluation.individualEvaluation(image_path,64,64,model)
print("# Predicted score is %s" % score)
