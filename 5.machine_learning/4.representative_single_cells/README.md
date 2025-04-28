# Representative single cells

In this module, we generate representative single-cell crops for Plate 3.
We use the model to get predicted probabilities, and randomly select cells based on if the cell is highly confident to be from a failing heart or a healthy heart.
We can generate these single-cell representation per perturbation or cell type.

In this case, we are generating three single-cell images, each representing:

1. Cell from a healthy heart with DMSO treatment
2. Cell from a failing heart with DMSO treatment
3. Cell from a failing heart with drug_x treatment
