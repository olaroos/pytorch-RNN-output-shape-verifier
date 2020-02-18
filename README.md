### README

this is a simple ipython file that verifies the way to split the output from pytorchs RNN modules. 

output, hidden = rnn.forward(input, hidden)

**Batch-First == True**</br>
outputs = outputs.view(bs,      seqlen,   ndirects, hidden_size)</br>
hidden  =  hidden.view(nlayers, ndirects, bs,       hidden_size)</br>

**Batch-First == False**</br>
outputs = outputs.view(seqlen,  bs,       ndirects, hidden_size)</br>
hidden  =  hidden.view(nlayers, ndirects, bs,       hidden_size)</br>

**view-splitting outputs incorrectly will throw an error if seqlen > 1**</br>
**view-splitting hidden incorrectly will not throw an error**
