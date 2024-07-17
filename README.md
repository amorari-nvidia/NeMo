 **Salm model export to ONNX**

The purpose of this branch is to export the Salm model (NeMo/examples/multimodal/speech_llm) to ONNX. Currently, the code exports the Encoder and the Modality adapter components.

 **Instructions**

Clone and switch to the branch:

```shell
git clone git@github.com:amorari-nvidia/NeMo.git
cd NeMo
git checkout salm-export

```

Run the container from the folder containing the /NeMo folder:

```shell
cd ..
docker run --gpus all -it -v .:/workspace  --shm-size=8g \
 --ulimit memlock=-1 --ulimit \
  stack=67108864 --device=/dev/snd  --name salm-export-nvcr-nemo-24-05 nvcr.io/nvidia/nemo:24.05
```

Reinstall NeMo:

```shell
pip uninstall nemo_toolkit
cd NeMo
./reinstall.sh
```

Download NeMo model checkpoint in the speech_llm folder:
```shell
cd examples/multimodal/speech_llm
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/nemo/speechllm_fc_llama2_7b/1.23.1/files?redirect=true&path=speechllm_fc_llama2_7b.nemo' -O speechllm_fc_llama2_7b.nemo
```

Execute the export script in the container:
```shell
 ./export_to_onnx.sh
```

Two ONNX models should have been created in NeMo/examples/multimodal/speech_llm/onnx.
