Your job is to investigate whether the BOE topic models have performance complementarity in the same way that the existing topic models have.

You will be given:
- My paper where I show performance complementarity between the existing topic models
- The code that generates the plots and tables in the paper
- Access to the database that contains the performance data for the existing topic models and the BOE topic models

You will need to:
- Use a subagent to read the paper and understand how performance complementarity is defined and was shown. The paper can be found at `/home/darren/Documents/PhD/phd-doc/paper_03_sacai/sacai_2025.tex`
- Use a subagent to investigate the code that generates the plots and tables in the paper. It is in `src/python/pipelines/performance_metrics`
- Use a subagent to investigate the database that contains the performance data for the existing topic models and the BOE topic models
    - To access the database run `. .env` followed by `psql $DB_URI <whatever>`
    - The relevant tables are defined in `src/python/pipelines/boe_performance_comparison/schema.sql` and `src/python/pipelines/performance_metrics/schema.sql`
    - The data can be thought of as a multi-dimensional matrix where the dimensions are:
        - corpus
        - model
        - num_topics
        - metric
        - repeats
        - metric_value
- Use a subagent to build the necessary investigation code to answer the question
- Your final answer should be placed in a folder called `src/python/pipelines/boe_performance_comparison/ignore/investigation` and should contain:
    - A latex file that explains the investigation and the results (performance complementarity or not). It should be equivalent to the results in the paper you have received
    - An `images` folder that contains the images that support your investigation
    - A markdown file that explains what you have done and where the code is, etc. (It is different from the latex file in that it is more detailed and explains the code in more detail)

Please note:
- In the same way that BERTopic with openai embeddings is different from BERTopic with sbert embeddings, different BOE embedding configurations generate different embeddings, which in turn generate different topic models
- There is already some code in `src/python/pipelines/boe_performance_comparison` that generates the plots, these may or may not be useful for your investigation
- To my best understanding, there are two BOE embedding configurations that differ only in the padding method: `knn_mean` or `noise_only`; however, I could be wrong
- You should use subagents whenever you think it is appropriate to do so, so as to keep the main agent focused on the task at hand
- You should plan how you want to conduct the investigation before you start in `src/python/pipelines/boe_performance_comparison/ignore/investigation/plan.md` and update it as you go along
- You should use uv for dependency management and to run the code