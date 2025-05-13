import os
import time
import subprocess


def mrm_process_commands(mrm: str):
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        f"Qwen/Qwen2.5-{mrm}B-Instruct-AWQ",
        "--host",
        "0.0.0.0",
        "--port",
        "30001",
        "--mem-fraction-static",
        "0.9",
        "--dp",
        "2",
        "--schedule-conservativeness",
        "0.35",
        "--attention-backend",
        "flashinfer",
        "--enable-torch-compile",
    ]

    env = dict(**os.environ, CUDA_VISIBLE_DEVICES="2,3")
    return cmd, env


if __name__ == "__main__":
    trl_dir = "/home/elicer/Development/trl"
    base_cmd = [
        "accelerate",
        "launch",
        "--config_file",
        f"{trl_dir}/examples/accelerate_configs/deepspeed_zero2_2gpus.yaml",
        f"{trl_dir}/examples/scripts/mpo.py",
    ]

    base_args = {
        "--dataset_name": "essay_writing",
        "--task_name": "essay_writing",
        "--wandb_entity": "iterater",
        "--wandb_project": "mpo-new",
        "--init_rm_prompt": None,
        "--learning_rate": "3e-6",
        "--num_ppo_epochs": "4",
        "--num_mpo_interval": None,
        "--num_mpo_samples": "20",
        "--save_n_updates": "20",
        "--num_mini_batches": "1",
        "--per_device_train_batch_size": "4",
        "--gradient_accumulation_steps": "8",
        "--local_rollout_forward_batch_size": "48",
        "--total_episodes": "26013",
        "--model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "--sft_model_path": "Qwen/Qwen2.5-3B-Instruct",
        "--response_length": "400",
        "--missing_eos_penalty": "1.0",
        "--kl_coef": "0.02",
        "--stop_token": "eos",
        "--reward_model_address": "http://ymscfzegfxjbrdnk.tunnel.elice.io",
        "--meta_reward_model_address": None,
    }

    # task_names = ["essay_writing", "summarization", "math_reasoning", "ethical_reasoning"]
    task_name = "essay_writing"
    # rms = ["1.5", "3", "7", "14"]
    rm = "1.5"
    mrms = ["3", "1.5", "7", "14"]

    # Conduct MPO experiments
    for mrm in mrms:
        print(f"Running MPO with RM: {rm} and MRM: {mrm}")
        if rm == mrm:
            meta_reward_model_address = "http://ymscfzegfxjbrdnk.tunnel.elice.io"
        else:
            mrm_cmd, mrm_env = mrm_process_commands(mrm)
            mrm_proc = subprocess.Popen(mrm_cmd, env=mrm_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"Launched meta RM process with PID: {mrm_proc.pid}")
            meta_reward_model_address = "http://0.0.0.0:30001"
            time.sleep(60 * 5)

        exp_variant = {
            "--exp_name": f"ew-mpo-{rm}_{mrm}",
            "--output_dir": f"{trl_dir}/models/policy-3b/essay_writing/mpo/{rm}_{mrm}",
            "--init_rm_prompt": f"{trl_dir}/trl/extras/mpo/prompts/essay_writing/evaluation_rubric_real_iter0.txt",
            "--num_mpo_interval": "10",
            "--meta_reward_model_address": meta_reward_model_address,
        }
        args = base_args.copy()
        args.update(exp_variant)
        full_cmd = ["CUDA_VISIBLE_DEVICES=0,1"] + base_cmd
        for key, value in args.items():
            full_cmd.extend([key, value])

        print("Launching:", " ".join(full_cmd))
        subprocess.run(" ".join(full_cmd), shell=True, check=True)
        if rm != mrm:
            print(f"Terminating meta reward model process with PID {mrm_proc.pid}")
            mrm_proc.terminate()
            time.sleep(60)
