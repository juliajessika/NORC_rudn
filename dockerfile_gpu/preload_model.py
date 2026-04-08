from deepcell.applications import NuclearSegmentation

print("Preloading DeepCell NuclearSegmentation model...")
app = NuclearSegmentation()
print("Done.")
print("model_mpp:", app.model_mpp)