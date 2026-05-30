Build out this timing pipeline in src/python/pipelines/timing. It should take a corpus and a topic model and return the time it takes to run the topic model on the corpus.
Look at the code in src/python/pipelines/topic_models/experiment_runner.py to see how to run the existing topic models.
See src/python/pipelines/boe_06_topic_models/experiment_runner.py to see how to run the BOE topic models.
We would also need to time how long it takes to make the boe embeddings; see src/python/pipelines/boe_04 & boe_05 for that.
Create a schema for the db table that will store this information.
We only need to do all of this for the three datasets: battery-abstracts, newsgroups, wikipedia-sample
However, it should be easy to add other datasets and set the number of repeats (initially 1)
Use subagents to design and implement.