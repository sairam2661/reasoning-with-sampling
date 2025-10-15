import pandas as pd
from grader_utils.he_grader import extract_code, entry_point



def fnames_to_json(data_file, fnames, output_fname, tag):
    
    output_file = output_fname + "_" + tag
    with open(output_file, "w") as fout:
        for fname in fnames:
            df = pd.read_csv(fname)
            for i in range(len(df)):
                task_id = df["id"][i]
                assert task_id == dataset[i]["task_id"]
                entry_point = dataset[i]["entry_point"]
                prompt = dataset[i]["prompt"]

                if tag="mcmc":
                    response = df["mcmc_completion"][i]
                elif tag="std":
                    response = prompt + df["std_completion"][i]
                elif tag="naive":
                    response = prompt + df["naive_completion"][i]

                code_completion = extract_code(response, entry_point)
      
                line = {
                    "task_id": df["id"][i],
                    "completion": code_completion,
                }
      
                fout.write(json.dumps(line) + "\n")
      


def he_results(data_file

