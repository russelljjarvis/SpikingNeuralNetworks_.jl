title: 'Towards Neuronal Deep Fakes: Data Driven Optimization of Reduced Neuronal Models'

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
Neuron models that behave like their biological counterparts are essential for computational neuroscience. Reduced neuron models simplify biological mechanisms in the interest of speed and interpretability. Only Reduced Neuron models can fit the scale of whole-brain simulations; therefore, improving these models is important. So far, little care has been taken to ensure that single neuron behaviours closely resemble biological neurons. To improve the veracity of reduced neuron models, I developed an optimizer that uses genetic algorithms to align model behaviours with those observed in experiments (first in Python, now in Julia). 

I verified that this optimizer could recover model parameters given only observed physiological data; however, I also found that reduced models nonetheless had limited ability to reproduce all observed behaviours and that this varied by cell type and desired behaviour. These challenges can be surmounted by carefully designing the set of physiological features that guide the optimization. In summary, we found evidence that reduced neuron model optimization had the potential to produce reduced neuron models for only a limited range of neuron types.

