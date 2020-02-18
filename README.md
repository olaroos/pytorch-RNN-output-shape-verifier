### README

this is a simple ipython file that verifies the way to split the output from pytorchs RNN modules. 

output, hidden = rnn.forward(input, hidden)

Batch-First == True 
outputs = outputs.view(bs,      seqlen,   ndirects, hidden_size)
hidden  =  hidden.view(nlayers, ndirects, bs,       hidden_size)

Batch-First == False 
outputs = outputs.view(seqlen,  bs,       ndirects, hidden_size)
hidden  =  hidden.view(nlayers, ndirects, bs,       hidden_size)

# If the sequence-length is larger than 1, pytorch will throw an error if outputs is view-split incorrectly.
# Pytorch will not throw an error when view-splitting the hidden-state. 
