from TD_Pipe import LLM, SamplingParams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    default="facebook/opt-125m",
)
parser.add_argument(
    "--pipeline-parallel-size",
    "-pp",
    type=int,
    default=1,
)
parser.add_argument(
    "--profiler-path",
    type=str,
    required=True,
)
args = parser.parse_args()

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM with specified configurations.
llm = LLM(
    model=args.model,
    pipeline_parallel_size=args.pipeline_parallel_size,
    profiler_path=args.profiler_path,
)

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
