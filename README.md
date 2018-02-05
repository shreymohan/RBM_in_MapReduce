# Restricted Boltzman Machine in MapReduce

* This was an attempt to integrate the restricted boltazman machine algorithm inside a MapReduce program.
* The training is split into two parts:
* The parameters delta_W, delta_hbias and delta_vbias are computed inside the mapper which would update the initialized Weights, hidden biases and Visible biases.
* Once computed, they are passed to the reducer through a custom writable class.
* Finally, the updation step is done in the reducer.
* This split in the training process is done in order to reduce the training time of the algorithm which may sometimes take a lot of time and computation.
* ISSUE - the code runs fine, the part files are being generated but the parameters are not changing/updating in every iteration. They remain the same.
