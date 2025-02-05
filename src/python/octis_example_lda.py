from octis.dataset.dataset import Dataset
from octis.models.LDA import LDA

dataset = Dataset()
dataset.fetch_dataset("20NewsGroup")


model = LDA(num_topics=20)  # Create model
model_output = model.train_model(dataset) # Train the model

# Get the topics
for topic in model_output['topics']:
    print(topic)
