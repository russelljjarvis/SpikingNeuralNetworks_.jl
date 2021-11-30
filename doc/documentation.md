title: 'Data Driven Optimization of Reduced Neuronal Models'

tags:
  - data driven optimization
  - reduced neuronal models
  - applied genetic algorithms
  - Julia
  - Metahieuristics

authors:
  - name: Russell Jarvis
    affiliation: Previous PhD Neuroscience, Arizona State University
  - name: Rick Gerkin

date: March 2021

Neuron models that behave like their biological counterparts are essential for computational neuroscience.
Reduced neuron models, which abstract away biological mechanisms in the interest of speed and interpretability, have received much attention due to their utility in large scale simulations of the brain, but little care has been taken to ensure that these models exhibit behaviors that closely resemble real neurons.
In order to improve the verisimilitude of these reduced neuron models, I developed an optimizer that uses genetic algorithms to align model behaviors with those observed in experiments.
I verified that this optimizer was able to recover model parameters given only observed physiological data; however, I also found that reduced models nonetheless had limited ability to reproduce all observed behaviors, and that this varied by cell type and desired behavior.
These challenges can partly be surmounted by carefully designing the set of physiological features that guide the optimization. In summary, we found evidence that reduced neuron model optimization had the potential to produce reduced neuron models for only a limited range of neuron types.

![Flowchart (2).jpg](Flowchart (2).jpg)
