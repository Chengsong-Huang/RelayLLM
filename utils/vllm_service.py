import argparse
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from vllm import LLM, SamplingParams
import torch
# No tokenizer needed on the server for this logic

# --- Global Variables ---
llm_model = None

# --- API Resource Definition ---
class GenerationResource(Resource):
    def post(self):
        if llm_model is None:
            return {'error': 'Model is not loaded yet'}, 503

        try:
            # Get the JSON data, which should be a list of requests
            payloads = request.get_json()
            
            if not isinstance(payloads, list):
                return {'error': 'Request body must be a list of generation requests'}, 400
            
            if not payloads:
                # Return the new expected response format
                return {'results': []}, 200

            # --- Create per-request SamplingParams ---
            # This is the correct way to handle batching with different params
            prompts = []
            individual_sampling_params = []
            
            for p in payloads:
                prompts.append(p['prompt'])
                individual_sampling_params.append(
                    SamplingParams(
                        max_tokens=p.get('max_tokens', 1024),
                        temperature=p.get('temperature', 0.7),
                        top_p=p.get('top_p', 1.0),
                        stop=p.get('stop', [])
                        # Add any other params you need (e.g., top_k)
                    )
                )

            # --- Perform batch inference ---
            # vLLM beautifully handles running a batch with different params
            outputs = llm_model.generate(prompts, individual_sampling_params)

            # --- Package results with finish_reason ---
            # We no longer do any truncation. We send exactly what vLLM produced
            # and tell the client *why* it stopped.
            final_results = []
            for output in outputs:
                # vLLM gives one output in the list per prompt
                sequence_output = output.outputs[0]
                final_results.append({
                    "text": sequence_output.text,
                    "finish_reason": sequence_output.finish_reason 
                    # This will be "stop" (hit EOS) or "length" (hit max_tokens)
                })
            print(payloads[0])
            print(final_results[0])
            # Return the new structured response
            return {'results': final_results}, 200

        except Exception as e:
            # Capture and return any errors
            return {'error': f'An error occurred: {str(e)}'}, 500

# --- Main Program Entry ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch-capable VLLM Flask Service")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the VLLM model")
    parser.add_argument('--port', type=int, default=7777, help="Port to run the service on")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Number of GPUs for tensor parallelism.")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help="Fraction of GPU memory to be used.")
    parser.add_argument('--max_num_seqs', type=int, default=256, help="Maximum number of sequences in a batch.")
    parser.add_argument('--trust_remote_code', action='store_true', help="Trust remote code for models")
    
    cli_args = parser.parse_args()

    # --- Model Loading ---
    print("="*50)
    print(f"Starting Batch VLLM Flask Server on port {cli_args.port}")
    print(f"Loading model: {cli_args.model_path}")
    
    # 加载模型
    llm_model = LLM(
        model=cli_args.model_path,
        tensor_parallel_size=cli_args.tensor_parallel_size,
        trust_remote_code=cli_args.trust_remote_code,
        gpu_memory_utilization=cli_args.gpu_memory_utilization,
        max_num_seqs=cli_args.max_num_seqs
    )
    print("Model loaded successfully!")
    print("Tokenizer is NOT loaded on server (not needed for this logic).")
    print("="*50)

    # --- Flask App Setup ---
    app = Flask(__name__)
    api = Api(app)
    api.add_resource(GenerationResource, '/generate')

    print(f"Flask server is ready. Send POST requests to http://0.0.0.0:{cli_args.port}/generate")
    app.run(host='0.0.0.0', port=cli_args.port, debug=False)
