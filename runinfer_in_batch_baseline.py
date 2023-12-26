import os
import sys
import subprocess

def run(decibel,mode_name,model_file, which_set):
    if which_set == "vbdm":
        tempfile_contents = f"""
        python infer_vanilla.py \
            +noisy_dir=data/vbdm/16384_clipped_{decibel}dB \
            +noisy_json= \
            +load_from={mode_name}/checkpoint.th \
            +out_dir=inferred/{decibel}dB/{mode_name} \
            +experiment=setting_3 \
            +model={model_file}
        """
    elif which_set == "dns":
        tempfile_contents = f"""
        python infer_vanilla.py \
            +noisy_dir=data/dns/16384_set_{decibel} \
            +noisy_json= \
            +load_from={mode_name}/checkpoint.th \
            +out_dir=inferred_dns/{decibel}dB/{mode_name} \
            +experiment=setting_3 \
            +model={model_file}
        """ 
    else:
        raise ValueError(f"no set {which_set}")
    with open("temp.sh",'w') as file:
        file.write(tempfile_contents)
    print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n{tempfile_contents}")
    subprocess.run("bash temp.sh",shell=True)

if __name__=="__main__":
    decibels = [1,3,7,15]
    modes_and_models = [
        ("realbaseline","realbaseline")
    ]
    for decibel in decibels:
        for mode_name, model_file in modes_and_models:
            run(decibel, mode_name, model_file, sys.argv[1])