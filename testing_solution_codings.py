## 1D
from solution_coding.descriptors import generate_molecular_descriptors
print("Molecular Descriptors:")
generate_molecular_descriptors
md = generate_molecular_descriptors("COc1ccc2[nH]c(C)c(/C=C3\Oc4cc(O)cc(O)c4C3=O)c2c1")
print(md)

## 2D
from solution_coding.maccs import generate_maccs_keys
print("MACCS Keys:")
mk = generate_maccs_keys("COc1ccc2[nH]c(C)c(/C=C3\Oc4cc(O)cc(O)c4C3=O)c2c1")
print(mk)

from solution_coding.morgan import generate_morgan_fingerprint
print("Morgan Fingerprints:")
mf = generate_morgan_fingerprint("COc1ccc2[nH]c(C)c(/C=C3\Oc4cc(O)cc(O)c4C3=O)c2c1")
print(mf)

from solution_coding.embedding import generate_word_embedding
print("Word Embeddings:")
we = generate_word_embedding("COc1ccc2[nH]c(C)c(/C=C3\Oc4cc(O)cc(O)c4C3=O)c2c1")
print(we)
