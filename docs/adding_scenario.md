# Creating a new Scenario

A scenario is an predefined method of testing an LLMs abilities. Users can define scenarios in the evaluation config.

Shown below is a config item for the `Instruction Following` scenario.

```yaml
  - name: Instruction Following
    id: inst
    desc: This scenario tests the model's ability to follow instructions on a range of different tasks.
    score_min: 0
    score_max: 10
    datasets: ["dim/mt_bench_en", "cnn_dailymail","xsum"]
    metrics:
    - name: Coherence
      desc: "Evaluate how well the response flows logically and how well ideas are connected. A high score should be given to responses that are easy to follow."

    - name: Completeness
      desc: "Consider whether the response adequately addresses the user's query, providing all necessary information. High scores should go to responses that are comprehensive."

    - name: Relevance
      desc: "Assess whether the response is directly related to the user's question and does not veer off-topic. High scores should be for highly relevant responses."

    - name: Depth
      desc: "Evaluate the extent to which the response delves into the topic, providing valuable insights or additional information. High scores should go to in-depth responses."

    - name: Creativity
      desc: "Consider whether the response displays creative thinking or novel approaches. High scores should be given to responses that demonstrate creativity when appropriate."

    - name: Level of Detail
      desc: "Examine whether the response offers a sufficient level of detail, catering to the user's needs. High scores should be given to responses with rich detail when necessary."
```

Let's break down each of the options, and see what they do.

* name: is a friendly name for the scenario to display in the web UI
* id: is the internal unique identifier for the scenario
* desc: a short description for the scenario which will be displayed in the web UI
* score_min / score_max: the range of values which the score for each metric can have
* datasets: a list of datasets to use in this scenario. Each dataset can contain one or more tasks (as defined in the dataset config) - which control how the data will be extracted and transformed into an input prompt for a model. All of these will be used in the scenario.
* metrics: a list of metrics to use to evaluate the response of a model given prompts from tasks sourced from each of the datasets. Any number of metrics may be defined, and each require a unique name and short description.