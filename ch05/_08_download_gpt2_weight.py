from gpt_download import download_and_load_gpt2
settings,params = download_and_load_gpt2(model_size="124M",models_dir="models")

print("Settings:",settings)
print("Parameter dictionnary keys :",params.keys())

print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)