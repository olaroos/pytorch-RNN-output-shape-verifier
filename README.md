### README

this is a simple ipython file that verifies the way to split the output from pytorchs RNN modules. 

output, hidden = rnn.forward(input, hidden)

**Batch-First == True**</br>
outputs = outputs.view(bs,      seqlen,   ndirects, hidden_size)</br>
hidden  =  hidden.view(nlayers, ndirects, bs,       hidden_size)</br>

**Batch-First == False**</br>
outputs = outputs.view(seqlen,  bs,       ndirects, hidden_size)</br>
hidden  =  hidden.view(nlayers, ndirects, bs,       hidden_size)</br>

**If the sequence-length is larger than 1, pytorch will throw an error if outputs is view-split incorrectly**</br>
**Pytorch will not throw an error when view-splitting the hidden-state**
