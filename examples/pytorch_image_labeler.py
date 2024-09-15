from core.inference.pytorch import InferWithPytorch, ImagenetLabeler
from core.readers.files import ReadFilepaths


# Gather some files to run inference on
path_reader = ReadFilepaths()
files = path_reader.do('data/flowers-102-100/**/*.jpg', recursive=True)


# Create a Pytorch Inference Action (and make it run on Apple's Silicon GPUs
pytorch_infer = InferWithPytorch("PytorchInferMPS", "mps")
probs = pytorch_infer.do(paths=files)

# Create a helper object to print predicted classes and their probabilities
labeler = ImagenetLabeler("dict/imagenet-simple-labels/imagenet-simple-labels.json")
top_classes, top_probs = labeler.top_n_classes(2, probs)
labels = labeler.labels_of_classes(top_classes, top_probs)

print(labels)

# ###

# Choose an image to pass through the model
# test_image = './../data/flowers-102-100/jpg/image_0001.jpg'
# probs = pytorch_infer.single_inference(image_path=test_image)

