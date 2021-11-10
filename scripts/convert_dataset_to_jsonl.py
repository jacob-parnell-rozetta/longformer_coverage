import json

with open("train.src") as f:
    source = f.readlines()
with open("train.tgt") as f:
    target = f.readlines()

with open("train.jsonl", "w") as outfile:
    for ix, val in enumerate(source):
        # Note: newline characters \n have since been replaced with NEWLINE_CHAR
        out_dict = {'document': val, 'summary': target[ix]}
        outfile.write(json.dumps(out_dict))
        outfile.write('\n')
